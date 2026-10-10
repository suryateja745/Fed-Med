"""
Global Model Checkpointing, Versioning, and State Management for FedMed Server.
Manages global 3D U-Net checkpoint serialization, best model tracking, round-level
versioning, persistent metadata export (global_model_metadata.json), and authenticated
global model weight encryption (AES-256-GCM / HMAC-SHA256).
"""

from datetime import datetime, timezone
import hashlib
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
from federation.security.global_encryption import (
    CryptographicIntegrityError,
    EncryptionMetadata,
    GlobalKeyManager,
    GlobalWeightEncryptionManager,
)
from federation.utils.config_loader import load_config
from federation.utils.logger import setup_logger


class GlobalModelManager:
    """
    Centralized coordinator model manager for the FedMed server.

    Responsibilities:
    - Save global model checkpoints after each federated aggregation round (global_model_round_{N}.pth).
    - Encrypt global model weights at rest (AES-256-GCM authenticated cipher with HMAC-SHA256 signatures).
    - Track and save historical best global model (best_global_model.pth / best_global_model.pth.enc).
    - Maintain latest global model pointer (global_model_latest.pth / global_model_latest.pth.enc).
    - Export and update global model metadata registry JSON (global_model_metadata.json).
    - Provide secure loading, integrity verification, and key rotation functions.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "./checkpoints",
        model: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        export_best: bool = True,
        keep_last_n: int = 5,
        enable_encryption: bool = True,
        encryption_manager: Optional[GlobalWeightEncryptionManager] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or load_config()
        self.model = model or build_unet3d_from_config(self.config)
        self.export_best = export_best
        self.keep_last_n = max(1, keep_last_n)

        self.logger = setup_logger(name="GlobalModelManager")
        self.metadata_file = self.checkpoint_dir / "global_model_metadata.json"

        # Initialize Global Model Weight Encryption Manager
        self.enable_encryption = enable_encryption
        if self.enable_encryption:
            if encryption_manager is not None:
                self.encryption_manager = encryption_manager
            else:
                key_path = self.checkpoint_dir / "fedmed_global_model.key"
                self.encryption_manager = GlobalWeightEncryptionManager(key_path=key_path)
            self.logger.info(
                f"[ModelManager] Global Model Weight Encryption ACTIVE "
                f"(Cipher: AES-256-GCM, Key ID: {self.encryption_manager.key_id})"
            )
        else:
            self.encryption_manager = None

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
            "encryption": {
                "enabled": self.enable_encryption,
                "cipher": "AES-256-GCM" if self.enable_encryption else "NONE",
                "key_id": self.encryption_manager.key_id if self.encryption_manager else None,
            },
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
        Save global model weights for a completed federated round, with automatic
        AES-256-GCM authenticated encryption and HMAC-SHA256 signature generation.

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
        dice_score = float(metrics_dict.get("val_dice_mean", metrics_dict.get("dice_score", -1.0)))

        # Paths
        round_path = self.checkpoint_dir / f"global_model_round_{round_num:03d}.pth"
        latest_path = self.checkpoint_dir / "global_model_latest.pth"

        # Save standard checkpoints
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
        is_best = False
        best_path = self.checkpoint_dir / "best_global_model.pth"
        if self.export_best and dice_score > self.best_dice:
            prev_best = self.best_dice
            self.best_dice = dice_score
            self.best_round = round_num
            is_best = True
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

        # Authenticated Encryption at Rest
        encryption_telemetry: Dict[str, Any] = {"enabled": False}
        if self.enable_encryption and self.encryption_manager:
            round_enc_path = self.encryption_manager.encrypt_checkpoint_file(
                source_path=round_path,
                round_num=round_num,
                dice_score=dice_score if dice_score >= 0 else None,
            )
            latest_enc_path = self.encryption_manager.encrypt_checkpoint_file(
                source_path=latest_path,
                target_path=self.checkpoint_dir / "global_model_latest.pth.enc",
                round_num=round_num,
                dice_score=dice_score if dice_score >= 0 else None,
            )
            if is_best:
                self.encryption_manager.encrypt_checkpoint_file(
                    source_path=best_path,
                    target_path=self.checkpoint_dir / "best_global_model.pth.enc",
                    round_num=round_num,
                    dice_score=dice_score if dice_score >= 0 else None,
                )

            # Compute sha256 checksum of encrypted checkpoint
            with open(round_enc_path, "rb") as f:
                enc_bytes = f.read()
            sha256_hash = hashlib.sha256(enc_bytes).hexdigest()

            encryption_telemetry = {
                "enabled": True,
                "cipher": "AES-256-GCM",
                "key_id": self.encryption_manager.key_id,
                "encrypted_round_checkpoint": str(round_enc_path),
                "encrypted_latest_checkpoint": str(latest_enc_path),
                "sha256": sha256_hash,
                "hmac_sha256": enc_bytes[-32:].hex(),
            }
            self.logger.info(
                f"[ModelManager] Authenticated encryption applied to Round {round_num} "
                f"(SHA256: {sha256_hash[:12]}..., Key ID: {self.encryption_manager.key_id})"
            )

        # Update metadata registry
        registry = self.get_metadata()
        registry["latest_round"] = round_num
        registry["latest_checkpoint"] = str(latest_path)
        if self.enable_encryption and self.encryption_manager:
            registry["encryption"] = {
                "enabled": True,
                "cipher": "AES-256-GCM",
                "key_id": self.encryption_manager.key_id,
                "encrypted_latest_checkpoint": str(self.checkpoint_dir / "global_model_latest.pth.enc"),
                "encrypted_best_checkpoint": str(self.checkpoint_dir / "best_global_model.pth.enc") if is_best or self.best_checkpoint_path else None,
            }

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
            "encryption": encryption_telemetry,
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
            if self.enable_encryption and self.encryption_manager:
                self.encryption_manager.encrypt_checkpoint_file(
                    source_path=best_path,
                    target_path=self.checkpoint_dir / "best_global_model.pth.enc",
                    round_num=round_num,
                    dice_score=dice_score,
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
        Transparently decrypts if loading from encrypted .enc checkpoint.
        """
        target_model = model or self.model
        latest_path = self.checkpoint_dir / "global_model_latest.pth"
        latest_enc_path = self.checkpoint_dir / "global_model_latest.pth.enc"

        if latest_path.exists():
            ckpt = load_model_checkpoint(model=target_model, filepath=latest_path, device=device)
            return target_model, ckpt

        if latest_enc_path.exists() and self.encryption_manager:
            return self.load_encrypted_checkpoint(latest_enc_path, model=target_model, device=device)

        round_ckpts = self.list_round_checkpoints()
        if round_ckpts:
            latest_path = round_ckpts[-1]
            ckpt = load_model_checkpoint(model=target_model, filepath=latest_path, device=device)
            return target_model, ckpt

        raise FileNotFoundError(f"No global checkpoint found in {self.checkpoint_dir}")

    def load_best(
        self,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load weights from best_global_model.pth into model.
        Transparently decrypts if loading from encrypted .enc checkpoint.
        """
        target_model = model or self.model
        best_path = self.checkpoint_dir / "best_global_model.pth"
        best_enc_path = self.checkpoint_dir / "best_global_model.pth.enc"

        if best_path.exists():
            ckpt = load_model_checkpoint(model=target_model, filepath=best_path, device=device)
            return target_model, ckpt

        if best_enc_path.exists() and self.encryption_manager:
            return self.load_encrypted_checkpoint(best_enc_path, model=target_model, device=device)

        raise FileNotFoundError(f"No best global checkpoint found at {best_path}")

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
        round_enc_path = self.checkpoint_dir / f"global_model_round_{round_num:03d}.pth.enc"

        if round_path.exists():
            ckpt = load_model_checkpoint(model=target_model, filepath=round_path, device=device)
            return target_model, ckpt

        if round_enc_path.exists() and self.encryption_manager:
            return self.load_encrypted_checkpoint(round_enc_path, model=target_model, device=device)

        raise FileNotFoundError(f"Round checkpoint not found: {round_path}")

    def load_encrypted_checkpoint(
        self,
        filepath: Union[str, Path],
        model: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Decrypt an encrypted .pth.enc checkpoint bundle in-memory and inject parameters into model.
        Validates HMAC signature and AES-GCM authentication tag before applying parameters.
        """
        if not self.encryption_manager:
            raise ValueError("Cannot load encrypted checkpoint: encryption manager is disabled.")

        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Encrypted checkpoint not found: {p}")

        target_model = model or self.model
        with open(p, "rb") as f:
            bundle_bytes = f.read()

        state_dict, metadata = self.encryption_manager.decrypt_state_dict(bundle_bytes, device=device)
        target_model.load_state_dict(state_dict)

        ckpt_info = {
            "round": metadata.round_num,
            "metrics": {"dice_score": metadata.dice_score},
            "encryption": {
                "verified": True,
                "cipher": metadata.cipher,
                "key_id": metadata.key_id,
            },
        }
        self.logger.info(
            f"[ModelManager] Successfully decrypted & verified encrypted checkpoint: {p.name} "
            f"(Round {metadata.round_num}, Key ID: {metadata.key_id})"
        )
        return target_model, ckpt_info

    def verify_checkpoint_integrity(
        self,
        checkpoint_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Verify the cryptographic integrity of an encrypted checkpoint file on disk.
        """
        if not self.encryption_manager:
            return {"valid": False, "error": "Encryption manager is disabled."}
        return self.encryption_manager.verify_bundle_integrity(checkpoint_path)

    def rotate_encryption_key(
        self,
        new_key: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Rotate the master encryption key and re-encrypt latest/best checkpoints.
        """
        if not self.encryption_manager:
            raise ValueError("Encryption manager is not active.")

        old_id = self.encryption_manager.key_id
        _, new_id = self.encryption_manager.key_manager.rotate_key(new_key=new_key)

        # Re-encrypt latest and best checkpoints with new key
        latest_pth = self.checkpoint_dir / "global_model_latest.pth"
        if latest_pth.exists():
            self.encryption_manager.encrypt_checkpoint_file(
                source_path=latest_pth,
                target_path=self.checkpoint_dir / "global_model_latest.pth.enc",
            )
        best_pth = self.checkpoint_dir / "best_global_model.pth"
        if best_pth.exists():
            self.encryption_manager.encrypt_checkpoint_file(
                source_path=best_pth,
                target_path=self.checkpoint_dir / "best_global_model.pth.enc",
            )

        # Update metadata registry
        registry = self.get_metadata()
        if "encryption" in registry:
            registry["encryption"]["key_id"] = new_id
            registry["encryption"]["last_rotated"] = datetime.now(timezone.utc).isoformat()
            self._write_metadata(registry)

        self.logger.info(f"[ModelManager] Rotated encryption key: {old_id} -> {new_id}")
        return {
            "status": "KEY_ROTATED",
            "old_key_id": old_id,
            "new_key_id": new_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def list_round_checkpoints(self) -> List[Path]:
        """Return sorted list of round-specific checkpoint files."""
        return sorted(list(self.checkpoint_dir.glob("global_model_round_*.pth")))

    def list_encrypted_checkpoints(self) -> List[Path]:
        """Return sorted list of encrypted round-specific checkpoint files."""
        return sorted(list(self.checkpoint_dir.glob("global_model_round_*.pth.enc")))

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

        # Also cleanup old encrypted checkpoints
        enc_files = self.list_encrypted_checkpoints()
        if len(enc_files) > self.keep_last_n:
            to_delete_enc = enc_files[:-self.keep_last_n]
            for f in to_delete_enc:
                try:
                    f.unlink()
                except Exception:
                    pass
