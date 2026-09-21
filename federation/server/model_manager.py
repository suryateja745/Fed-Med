"""
Global Model Checkpointing, Versioning, and State Management for FedMed Server.
Manages global 3D U-Net checkpoint serialization, best model tracking, round-level
versioning, and persistent metadata export (global_model_metadata.json).
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from federation.models.unet3d import (
    build_unet3d_from_config,
    load_model_checkpoint,
    save_model_checkpoint,
    set_model_parameters,
)
from federation.utils.config_loader import load_config
from federation.utils.logger import setup_logger


class GlobalModelManager:
    """
    Centralized coordinator model manager for the FedMed server.

    Responsibilities:
    - Save global model checkpoints after each federated aggregation round (global_model_round_{N}.pth).
    - Track and save historical best global model (best_global_model.pth) based on validation Dice score.
    - Maintain latest global model pointer (global_model_latest.pth).
    - Export and update global model metadata registry JSON (global_model_metadata.json).
    - Provide retrieval and loading functions for model inference and deployment.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "./checkpoints",
        model: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        export_best: bool = True,
        keep_last_n: int = 5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or load_config()
        self.model = model or build_unet3d_from_config(self.config)
        self.export_best = export_best
        self.keep_last_n = max(1, keep_last_n)

        self.logger = setup_logger(name="GlobalModelManager")
        self.metadata_file = self.checkpoint_dir / "global_model_metadata.json"

        self.best_dice: float = -1.0
        self.best_round: int = 0
        self.best_checkpoint_path: Optional[Path] = None

        self._init_metadata_registry()

    def _init_metadata_registry(self) -> None:
        """Initialize or load existing metadata registry."""
        if self.metadata_file.exists():
            try:
                data = self.get_metadata()
                self.best_dice = float(data.get("best_dice", -1.0))
                self.best_round = int(data.get("best_round", 0))
                if data.get("best_checkpoint"):
                    self.best_checkpoint_path = Path(data["best_checkpoint"])
                return
            except Exception:
                pass

        initial_registry = {
            "project": "FedMed",
            "model_architecture": self.config.get("model", {}).get("name", "UNet3D"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "latest_round": 0,
            "best_round": 0,
            "best_dice": -1.0,
            "best_checkpoint": None,
            "latest_checkpoint": None,
            "rounds_history": [],
        }
        self._write_metadata(initial_registry)

    def _write_metadata(self, data: Dict[str, Any]) -> None:
        """Write metadata dictionary to JSON file."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to write model metadata: {e}")

    def get_metadata(self) -> Dict[str, Any]:
        """Read and return model metadata registry dictionary."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"rounds_history": []}

    def save_round_checkpoint(
        self,
        round_num: int,
        parameters: Optional[List[np.ndarray]] = None,
        model: Optional[nn.Module] = None,
        metrics: Optional[Dict[str, Any]] = None,
        participating_clients: Optional[List[str]] = None,
    ) -> Path:
        """
        Save global model weights for a completed federated round.

        Args:
            round_num: Current federated aggregation round number.
            parameters: Aggregated global parameter ndarrays (optional if model is passed).
            model: PyTorch model instance (optional if parameters are passed).
            metrics: Aggregated metrics dictionary for the round.
            participating_clients: List of client IDs that contributed to this round.

        Returns:
            Path to saved latest checkpoint file.
        """
        target_model = model or self.model
        if parameters is not None:
            set_model_parameters(target_model, parameters)

        metrics_dict = metrics or {}
        timestamp = datetime.now(timezone.utc).isoformat()

        # Paths
        round_path = self.checkpoint_dir / f"global_model_round_{round_num:03d}.pth"
        latest_path = self.checkpoint_dir / "global_model_latest.pth"

        # Save round-specific and latest checkpoints
        save_model_checkpoint(
            model=target_model,
            filepath=round_path,
            round_num=round_num,
            metrics=metrics_dict,
        )
        save_model_checkpoint(
            model=target_model,
            filepath=latest_path,
            round_num=round_num,
            metrics=metrics_dict,
        )

        # Check for new best model
        dice_score = float(metrics_dict.get("val_dice_mean", metrics_dict.get("dice_score", -1.0)))
        is_best = False
        if self.export_best and dice_score > self.best_dice:
            prev_best = self.best_dice
            self.best_dice = dice_score
            self.best_round = round_num
            is_best = True
            best_path = self.checkpoint_dir / "best_global_model.pth"
            save_model_checkpoint(
                model=target_model,
                filepath=best_path,
                round_num=round_num,
                metrics={**metrics_dict, "best_dice": dice_score},
            )
            self.best_checkpoint_path = best_path
            self.logger.info(
                f"[ModelManager] New best global model saved (Round {round_num})! "
                f"Dice: {prev_best:.4f} -> {dice_score:.4f}"
            )

        # Update metadata registry
        registry = self.get_metadata()
        registry["latest_round"] = round_num
        registry["latest_checkpoint"] = str(latest_path)
        if is_best:
            registry["best_round"] = round_num
            registry["best_dice"] = dice_score
            registry["best_checkpoint"] = str(self.best_checkpoint_path)

        round_entry = {
            "round": round_num,
            "timestamp": timestamp,
            "metrics": metrics_dict,
            "global_dice_score": dice_score if dice_score >= 0 else None,
            "participating_clients": participating_clients or [],
            "checkpoint_path": str(round_path),
            "is_best": is_best,
        }
        registry.setdefault("rounds_history", []).append(round_entry)
        self._write_metadata(registry)

        self._cleanup_old_round_checkpoints()
        self.logger.info(
            f"[ModelManager] Saved global checkpoint for Round {round_num} -> {round_path.name}"
        )

        return latest_path

    def update_evaluation_metrics(
        self,
        round_num: int,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Update registry with evaluation results for a specific round.
        """
        registry = self.get_metadata()
        dice_score = float(metrics.get("val_dice_mean", metrics.get("dice_score", -1.0)))

        is_best = False
        if self.export_best and dice_score > self.best_dice:
            prev_best = self.best_dice
            self.best_dice = dice_score
            self.best_round = round_num
            is_best = True

            best_path = self.checkpoint_dir / "best_global_model.pth"
            save_model_checkpoint(
                model=self.model,
                filepath=best_path,
                round_num=round_num,
                metrics={**metrics, "best_dice": dice_score},
            )
            self.best_checkpoint_path = best_path
            registry["best_round"] = round_num
            registry["best_dice"] = dice_score
            registry["best_checkpoint"] = str(best_path)
            self.logger.info(
                f"[ModelManager] New best global model evaluated (Round {round_num})! "
                f"Dice: {prev_best:.4f} -> {dice_score:.4f}"
            )

        # Update matching round in history
        for entry in registry.get("rounds_history", []):
            if entry.get("round") == round_num:
                entry.setdefault("metrics", {}).update(metrics)
                if dice_score >= 0:
                    entry["global_dice_score"] = dice_score
                if is_best:
                    entry["is_best"] = True
                break

        self._write_metadata(registry)

    def load_latest(
        self,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load weights from global_model_latest.pth into model.
        """
        target_model = model or self.model
        latest_path = self.checkpoint_dir / "global_model_latest.pth"
        if not latest_path.exists():
            round_ckpts = self.list_round_checkpoints()
            if round_ckpts:
                latest_path = round_ckpts[-1]
            else:
                raise FileNotFoundError(f"No global checkpoint found in {self.checkpoint_dir}")

        ckpt = load_model_checkpoint(model=target_model, filepath=latest_path, device=device)
        return target_model, ckpt

    def load_best(
        self,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load weights from best_global_model.pth into model.
        """
        target_model = model or self.model
        best_path = self.checkpoint_dir / "best_global_model.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"No best global checkpoint found at {best_path}")

        ckpt = load_model_checkpoint(model=target_model, filepath=best_path, device=device)
        return target_model, ckpt

    def load_round(
        self,
        round_num: int,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load weights from specific round checkpoint into model.
        """
        target_model = model or self.model
        round_path = self.checkpoint_dir / f"global_model_round_{round_num:03d}.pth"
        if not round_path.exists():
            raise FileNotFoundError(f"Round checkpoint not found: {round_path}")

        ckpt = load_model_checkpoint(model=target_model, filepath=round_path, device=device)
        return target_model, ckpt

    def list_round_checkpoints(self) -> List[Path]:
        """Return sorted list of round-specific checkpoint files."""
        return sorted(list(self.checkpoint_dir.glob("global_model_round_*.pth")))

    def _cleanup_old_round_checkpoints(self) -> None:
        """Enforce keep_last_n limit on intermediate round checkpoints."""
        round_files = self.list_round_checkpoints()
        if len(round_files) > self.keep_last_n:
            to_delete = round_files[:-self.keep_last_n]
            for f in to_delete:
                try:
                    f.unlink()
                except Exception:
                    pass
