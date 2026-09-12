"""
Automated Triggers and Dispatch Daemons for FedMed Server Coordinator.
Implements:
- AutoDispatchTrigger: Automatically serves the latest/best global preset model weights
  to connecting hospital clients in unattended/offline admin mode and records an audit log.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.models.unet3d import (
    build_unet3d_from_config,
    get_model_parameters,
    load_model_checkpoint,
)
from federation.server.model_manager import GlobalModelManager
from federation.utils.config_loader import load_config
from federation.utils.logger import setup_logger


@dataclass
class DispatchAuditEvent:
    """Audit record for an automated model dispatch event."""
    event_id: str
    hospital_id: str
    model_version: str
    checkpoint_path: str
    round_num: Optional[int]
    dice_score: Optional[float]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "SUCCESS"
    client_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutoDispatchTrigger:
    """
    Automated Model Dispatch Trigger for Unattended / Offline Admin Mode.

    Responsibilities:
    - Listen for hospital client registration or model weight pull requests.
    - Query global model checkpoint registry to find the highest-performing (best) model
      or latest aggregated global weights.
    - Automatically deserialize and stream model parameters to the requesting client.
    - Record structured, persistent audit logs (dispatch_audit.jsonl) for regulatory and compliance tracking.
    """

    def __init__(
        self,
        model_manager: Optional[GlobalModelManager] = None,
        checkpoint_dir: Union[str, Path] = "./checkpoints",
        audit_log_file: Union[str, Path] = "./logs/dispatch_audit.jsonl",
        prefer_best: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.audit_log_file = Path(audit_log_file)
        self.audit_log_file.parent.mkdir(parents=True, exist_ok=True)
        self.prefer_best = prefer_best
        self.config = config or load_config()

        self.logger = setup_logger(name="AutoDispatchTrigger")
        self.model_manager = model_manager or GlobalModelManager(
            checkpoint_dir=self.checkpoint_dir,
            config=self.config,
        )

        self._dispatch_count: int = 0
        self._served_hospitals: Dict[str, int] = {}

    def handle_client_connect(
        self,
        hospital_id: str,
        client_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Handle a hospital client connection or initial parameter request in offline admin mode.
        Resolves the best available global weights, logs the event, and returns parameters.

        Args:
            hospital_id: Hospital node identifier.
            client_info: Optional metadata from client (hardware, dataset size, OS, etc.).

        Returns:
            Tuple of (model_parameters_numpy_list, dispatch_metadata_dict).
        """
        client_meta = client_info or {}
        event_id = f"disp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{hospital_id}"

        # 1. Resolve Best or Latest Checkpoint
        model_weights, version_name, ckpt_path, round_num, dice_score = self._resolve_global_weights()

        # 2. Build and Record Audit Record
        audit_event = DispatchAuditEvent(
            event_id=event_id,
            hospital_id=hospital_id,
            model_version=version_name,
            checkpoint_path=str(ckpt_path),
            round_num=round_num,
            dice_score=dice_score,
            status="SUCCESS",
            client_info=client_meta,
        )
        self._record_audit_log(audit_event)

        # 3. Update internal counters
        self._dispatch_count += 1
        self._served_hospitals[hospital_id] = self._served_hospitals.get(hospital_id, 0) + 1

        self.logger.info(
            f"[AutoDispatch] Successfully served '{version_name}' (Round {round_num}, "
            f"Dice={dice_score if dice_score is not None else 'N/A'}) to '{hospital_id}' "
            f"[Total Dispatches: {self._dispatch_count}]."
        )

        metadata = {
            "event_id": event_id,
            "hospital_id": hospital_id,
            "model_version": version_name,
            "checkpoint_path": str(ckpt_path),
            "round_num": round_num,
            "dice_score": dice_score,
            "timestamp": audit_event.timestamp,
        }

        return model_weights, metadata

    def _resolve_global_weights(
        self,
    ) -> Tuple[List[np.ndarray], str, Path, Optional[int], Optional[float]]:
        """
        Resolve the most appropriate model weights:
        1. best_global_model.pth (if prefer_best and exists)
        2. global_model_latest.pth
        3. Fresh preset baseline model (if no rounds completed yet)
        """
        metadata = self.model_manager.get_metadata()
        best_path = self.checkpoint_dir / "best_global_model.pth"
        latest_path = self.checkpoint_dir / "global_model_latest.pth"

        # Check for best model
        if self.prefer_best and best_path.exists():
            try:
                loaded_model, ckpt = self.model_manager.load_best()
                params = get_model_parameters(loaded_model)
                best_round = int(ckpt.get("round", metadata.get("best_round", 0)))
                best_dice = float(metadata.get("best_dice", -1.0))
                return (
                    params,
                    "best_global_model",
                    best_path,
                    best_round,
                    best_dice if best_dice >= 0 else None,
                )
            except Exception as e:
                self.logger.warning(f"[AutoDispatch] Could not load best model ({e}), falling back to latest.")

        # Check for latest model
        if latest_path.exists():
            try:
                loaded_model, ckpt = self.model_manager.load_latest()
                params = get_model_parameters(loaded_model)
                latest_round = int(ckpt.get("round", metadata.get("latest_round", 0)))
                round_dice = float(ckpt.get("metrics", {}).get("val_dice_mean", -1.0))
                return (
                    params,
                    f"global_model_round_{latest_round:03d}",
                    latest_path,
                    latest_round,
                    round_dice if round_dice >= 0 else None,
                )
            except Exception as e:
                self.logger.warning(f"[AutoDispatch] Could not load latest model ({e}), falling back to preset baseline.")

        # Preset baseline initialized model
        params = get_model_parameters(self.model_manager.model)
        return (
            params,
            "preset_baseline_round_0",
            self.checkpoint_dir / "preset_baseline.pth",
            0,
            None,
        )


    def _record_audit_log(self, event: DispatchAuditEvent) -> None:
        """Append audit event JSON line to persistent audit log file."""
        try:
            with open(self.audit_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            self.logger.error(f"[AutoDispatch] Failed to write audit log: {e}")

    def get_dispatch_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read and return audit log entries from disk."""
        events: List[Dict[str, Any]] = []
        if not self.audit_log_file.exists():
            return events

        try:
            with open(self.audit_log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception as e:
            self.logger.error(f"[AutoDispatch] Error reading audit log: {e}")

        if limit is not None:
            return events[-limit:]
        return events

    def get_dispatch_stats(self) -> Dict[str, Any]:
        """Return summary statistics of automated model dispatches."""
        history = self.get_dispatch_history()
        return {
            "total_dispatches": len(history) if history else self._dispatch_count,
            "unique_hospitals_served": len(set(e.get("hospital_id") for e in history)) if history else len(self._served_hospitals),
            "hospitals_served_breakdown": self._served_hospitals,
            "last_dispatch": history[-1] if history else None,
            "audit_log_path": str(self.audit_log_file),
        }
