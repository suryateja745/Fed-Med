"""
Local Model Checkpointing and Parameter Diffing for FedMed Clients.
Manages local model backups, fallback recovery on training failure, best checkpoint
tracking, and weight delta computation (Delta W = W_local - W_global) for bandwidth optimization.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from federation.models.unet3d import (
    load_model_checkpoint,
    save_model_checkpoint,
    set_model_parameters,
)
from federation.utils.logger import setup_logger


# Parameter Diffing & Delta Utilities

def compute_parameter_delta(
    local_params: List[np.ndarray],
    global_params: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Compute weight updates (parameter deltas) between local and global models:
    Delta W = W_local - W_global

    Args:
        local_params: List of NumPy ndarrays from local model after training.
        global_params: List of NumPy ndarrays from server before local training.

    Returns:
        List of NumPy ndarrays representing parameter differences.
    """
    if len(local_params) != len(global_params):
        raise ValueError(
            f"Mismatched parameter list lengths: local has {len(local_params)}, "
            f"global has {len(global_params)}"
        )

    deltas = []
    for loc, glob in zip(local_params, global_params):
        if loc.shape != glob.shape:
            raise ValueError(f"Shape mismatch in parameter arrays: {loc.shape} vs {glob.shape}")
        
        # Preserve integer buffer types
        if np.issubdtype(loc.dtype, np.integer):
            deltas.append(loc - glob)
        else:
            deltas.append((loc - glob).astype(np.float32))

    return deltas


def apply_parameter_delta(
    base_params: List[np.ndarray],
    delta_params: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Reconstruct new parameters by applying parameter deltas:
    W_new = W_base + Delta W

    Args:
        base_params: Baseline parameter ndarrays.
        delta_params: Parameter delta ndarrays.

    Returns:
        Reconstructed list of NumPy ndarrays.
    """
    if len(base_params) != len(delta_params):
        raise ValueError(
            f"Mismatched parameter list lengths: base has {len(base_params)}, "
            f"delta has {len(delta_params)}"
        )

    updated = []
    for base, delta in zip(base_params, delta_params):
        if base.shape != delta.shape:
            raise ValueError(f"Shape mismatch: base {base.shape} vs delta {delta.shape}")
        
        if np.issubdtype(base.dtype, np.integer):
            updated.append(base + delta)
        else:
            updated.append((base + delta).astype(np.float32))

    return updated


def compute_delta_statistics(
    delta_params: List[np.ndarray],
    sparsity_threshold: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute summary statistics on parameter deltas (L2 norm, Linf norm, sparsity).

    Args:
        delta_params: List of parameter delta ndarrays.
        sparsity_threshold: Threshold below which updates are considered near-zero.

    Returns:
        Dictionary with 'l2_norm', 'max_abs_update', 'total_elements', 'sparsity_ratio'.
    """
    total_elements = sum(d.size for d in delta_params)
    if total_elements == 0:
        return {
            "l2_norm": 0.0,
            "max_abs_update": 0.0,
            "total_elements": 0,
            "sparsity_ratio": 1.0,
        }

    sum_sq = 0.0
    max_val = 0.0
    near_zero_count = 0

    for d in delta_params:
        if d.size == 0:
            continue
        abs_d = np.abs(d)
        sum_sq += float(np.sum(abs_d ** 2))
        max_val = max(max_val, float(np.max(abs_d)))
        near_zero_count += int(np.sum(abs_d < sparsity_threshold))

    l2_norm = float(np.sqrt(sum_sq))
    sparsity_ratio = float(near_zero_count / max(1, total_elements))

    return {
        "l2_norm": round(l2_norm, 6),
        "max_abs_update": round(max_val, 6),
        "total_elements": total_elements,
        "sparsity_ratio": round(sparsity_ratio, 4),
    }


# Client Checkpoint Manager

class ClientCheckpointManager:
    """
    Manages local model checkpointing, global parameter backups, best model saving,
    and automatic fallback recovery for hospital nodes during federated learning.
    """

    def __init__(
        self,
        hospital_id: str,
        checkpoint_dir: Union[str, Path] = "./checkpoints",
        keep_last_n: int = 3,
    ) -> None:
        self.hospital_id = str(hospital_id)
        self.checkpoint_dir = Path(checkpoint_dir) / self.hospital_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = max(1, keep_last_n)

        self.logger = setup_logger(name=f"Checkpoint-{self.hospital_id}")
        self.cached_global_parameters: Optional[List[np.ndarray]] = None

        self.best_dice: float = -1.0
        self.best_checkpoint_path: Optional[Path] = None

    def backup_global_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Store an in-memory backup of global model parameters received from server
        before beginning local training.
        """
        self.cached_global_parameters = [np.copy(p) for p in parameters]

    def restore_global_parameters(self, model: nn.Module) -> bool:
        """
        Restore model to the last backed-up global parameters if local training fails.

        Returns:
            True if backup was successfully restored, False if no backup exists.
        """
        if self.cached_global_parameters is None:
            self.logger.warning("No global parameters in cache to restore.")
            return False

        try:
            set_model_parameters(model, self.cached_global_parameters)
            self.logger.info(f"[{self.hospital_id}] Successfully restored model to global parameter baseline.")
            return True
        except Exception as e:
            self.logger.error(f"[{self.hospital_id}] Failed to restore global parameters: {e}")
            return False

    def save_latest(
        self,
        model: nn.Module,
        round_num: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save latest local model checkpoint and round-specific checkpoint.
        Rotates older round checkpoints if keep_last_n is exceeded.

        Returns:
            Path to saved latest checkpoint file.
        """
        latest_path = self.checkpoint_dir / f"{self.hospital_id}_latest.pth"
        round_path = self.checkpoint_dir / f"{self.hospital_id}_round_{round_num:03d}.pth"

        # Save latest checkpoint
        save_model_checkpoint(
            model=model,
            filepath=latest_path,
            optimizer=optimizer,
            round_num=round_num,
            metrics=metrics,
        )

        # Save round-specific checkpoint
        save_model_checkpoint(
            model=model,
            filepath=round_path,
            optimizer=optimizer,
            round_num=round_num,
            metrics=metrics,
        )

        self._cleanup_old_round_checkpoints()
        self.logger.info(f"[{self.hospital_id}] Saved local checkpoint for Round {round_num} -> {latest_path.name}")
        return latest_path

    def save_best(
        self,
        model: nn.Module,
        round_num: int,
        dice_score: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Save best checkpoint if current Dice score exceeds previous historical best.

        Returns:
            Path to best checkpoint file if improved, else None.
        """
        if dice_score > self.best_dice:
            prev_best = self.best_dice
            self.best_dice = dice_score
            best_path = self.checkpoint_dir / f"{self.hospital_id}_best.pth"

            save_model_checkpoint(
                model=model,
                filepath=best_path,
                round_num=round_num,
                metrics={**(metrics or {}), "best_dice": dice_score},
            )
            self.best_checkpoint_path = best_path
            self.logger.info(
                f"[{self.hospital_id}] New best model saved! Dice improved: {prev_best:.4f} -> {dice_score:.4f}"
            )
            return best_path
        return None

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load latest available checkpoint into model and optimizer.
        """
        latest_path = self.checkpoint_dir / f"{self.hospital_id}_latest.pth"
        if not latest_path.exists():
            # Check for any round checkpoints
            round_files = self.list_round_checkpoints()
            if round_files:
                latest_path = round_files[-1]
            else:
                raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")

        return load_model_checkpoint(model=model, filepath=latest_path, optimizer=optimizer, device=device)

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load historical best checkpoint into model and optimizer.
        """
        best_path = self.checkpoint_dir / f"{self.hospital_id}_best.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"No best checkpoint found at {best_path}")

        return load_model_checkpoint(model=model, filepath=best_path, optimizer=optimizer, device=device)

    def list_round_checkpoints(self) -> List[Path]:
        """Return sorted list of round-specific checkpoint file paths."""
        return sorted(list(self.checkpoint_dir.glob(f"{self.hospital_id}_round_*.pth")))

    def _cleanup_old_round_checkpoints(self) -> None:
        """Enforce keep_last_n limit by deleting oldest round checkpoints."""
        round_files = self.list_round_checkpoints()
        if len(round_files) > self.keep_last_n:
            to_delete = round_files[:-self.keep_last_n]
            for f in to_delete:
                try:
                    f.unlink()
                except Exception:
                    pass
