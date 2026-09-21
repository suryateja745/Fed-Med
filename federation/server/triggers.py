"""
Automated Triggers and Dispatch Daemons for FedMed Server Coordinator.
Implements:
- AutoDispatchTrigger: Automatically serves the latest/best global preset model weights
  to connecting hospital clients in unattended/offline admin mode and records an audit log.
- AutoAggregateTrigger: Automatically executes federated parameter aggregation (FedMedStrategy/FedAvg),
  updates global model checkpoints, increments the round counter, and notifies subscribers upon
  reaching the client upload threshold.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
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
    set_model_parameters,
)
from federation.server.model_manager import GlobalModelManager
from federation.server.strategy import (
    ClientProxy,
    FedMedStrategy,
    FitRes,
    aggregate_weighted_parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from federation.server.sync_manager import PendingClientUpdate, RoundSyncManager
from federation.utils.config_loader import load_config
from federation.utils.logger import setup_logger


# Data Structures & Event Models

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


@dataclass
class RoundCompleteEvent:
    """Event payload emitted when automated aggregation completes a federated round."""
    round_num: int
    participating_clients: List[str]
    num_samples_total: int
    metrics: Dict[str, Any]
    checkpoint_path: str
    is_new_best: bool
    duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# AutoDispatchTrigger

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


# AutoAggregateTrigger

class AutoAggregateTrigger:
    """
    Automated Parameter Aggregation and Model Publication Trigger.

    Responsibilities:
    - Listen for hospital client parameter uploads into RoundSyncManager.
    - Check if upload threshold (min_fit_clients) or timeout is satisfied.
    - Execute FedMedStrategy.aggregate_fit() to perform weighted averaging.
    - Update global model weights and save versioned checkpoint (global_model_round_X.pth).
    - Automatically increment active federated round in RoundSyncManager.
    - Dispatch RoundCompleteEvent notifications to registered subscribers/callbacks.
    """

    def __init__(
        self,
        sync_manager: RoundSyncManager,
        strategy: Optional[FedMedStrategy] = None,
        model_manager: Optional[GlobalModelManager] = None,
        min_upload_threshold: Optional[int] = None,
        auto_advance_round: bool = True,
        audit_log_file: Union[str, Path] = "./logs/aggregation_events.jsonl",
        on_round_complete_callbacks: Optional[List[Callable[[RoundCompleteEvent], None]]] = None,
    ) -> None:
        self.sync_manager = sync_manager
        self.min_upload_threshold = (
            min_upload_threshold
            if min_upload_threshold is not None
            else self.sync_manager.min_clients_per_round
        )
        self.model_manager = model_manager or getattr(strategy, "model_manager", None) or GlobalModelManager()
        self.strategy = strategy or FedMedStrategy(model_manager=self.model_manager)
        self.auto_advance_round = auto_advance_round

        self.audit_log_file = Path(audit_log_file)
        self.audit_log_file.parent.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(name="AutoAggregateTrigger")
        self._callbacks: List[Callable[[RoundCompleteEvent], None]] = list(on_round_complete_callbacks or [])
        self._completed_events: List[RoundCompleteEvent] = []

    def register_on_round_complete(self, callback: Callable[[RoundCompleteEvent], None]) -> None:
        """Register an event listener for round completion notifications."""
        self._callbacks.append(callback)

    def handle_client_upload(
        self,
        hospital_id: str,
        round_num: int,
        parameters: List[np.ndarray],
        num_examples: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, Optional[RoundCompleteEvent]]:
        """
        Handle a parameter upload from a hospital client node.
        Ingests update into synchronization queue and automatically evaluates aggregation trigger.

        Args:
            hospital_id: Client identifier.
            round_num: Client training round number.
            parameters: Trained model weight arrays.
            num_examples: Training sample count.
            metrics: Training telemetry dictionary.

        Returns:
            Tuple of (upload_success: bool, status_message: str, round_event: Optional[RoundCompleteEvent]).
        """
        ok, msg = self.sync_manager.submit_update(
            hospital_id=hospital_id,
            round_num=round_num,
            parameters=parameters,
            num_examples=num_examples,
            metrics=metrics,
        )
        if not ok:
            return False, msg, None

        # Check if threshold reached
        triggered, event = self.check_and_trigger(round_num=round_num)
        return True, msg, event

    def check_and_trigger(
        self,
        round_num: Optional[int] = None,
        force: bool = False,
    ) -> Tuple[bool, Optional[RoundCompleteEvent]]:
        """
        Evaluate whether the current round has enough uploads to execute automated aggregation.

        Args:
            round_num: Round to evaluate (defaults to sync_manager.current_round).
            force: If True, executes aggregation regardless of whether min threshold is reached.

        Returns:
            Tuple of (triggered: bool, event: Optional[RoundCompleteEvent]).
        """
        target_round = round_num if round_num is not None else self.sync_manager.current_round
        ready_updates = self.sync_manager.get_ready_updates(round_num=target_round)
        upload_count = len(ready_updates)

        if not force and upload_count < self.min_upload_threshold:
            return False, None

        start_time = time.time()
        self.logger.info(
            f"[AutoAggregate] Triggering aggregation for Round {target_round} "
            f"({upload_count}/{self.min_upload_threshold} hospital updates ready)..."
        )

        # 1. Package Client Results for Aggregation
        fit_results = []
        for upd in ready_updates:
            proxy = ClientProxy(cid=upd.hospital_id)
            param_obj = ndarrays_to_parameters(upd.parameters)
            fit_res = FitRes(
                parameters=param_obj,
                num_examples=upd.num_examples,
                metrics=upd.metrics,
            )
            fit_results.append((proxy, fit_res))

        # 2. Execute Strategy Aggregation
        agg_params_obj, agg_metrics = self.strategy.aggregate_fit(
            server_round=target_round,
            results=fit_results,
            failures=[],
        )

        # Convert back to NumPy ndarrays
        if agg_params_obj is not None:
            aggregated_ndarrays = parameters_to_ndarrays(agg_params_obj)
        else:
            # Fallback direct weighted average
            weighted_inputs = [
                (upd.parameters, float(upd.num_examples) * (1.0 + float(upd.metrics.get("val_dice_mean", 0.0))))
                for upd in ready_updates
            ]
            aggregated_ndarrays = aggregate_weighted_parameters(weighted_inputs)

        # 3. Update Model State & Checkpoint
        set_model_parameters(self.model_manager.model, aggregated_ndarrays)

        # Calculate summary metrics
        total_samples = sum(upd.num_examples for upd in ready_updates)
        participating_clients = [upd.hospital_id for upd in ready_updates]
        avg_train_loss = float(np.mean([upd.metrics.get("train_loss", 1.0) for upd in ready_updates]))
        avg_dice_mean = float(np.mean([upd.metrics.get("val_dice_mean", 0.0) for upd in ready_updates]))
        avg_dice_tc = float(np.mean([upd.metrics.get("val_dice_tc", 0.0) for upd in ready_updates]))
        avg_dice_wt = float(np.mean([upd.metrics.get("val_dice_wt", 0.0) for upd in ready_updates]))
        avg_dice_et = float(np.mean([upd.metrics.get("val_dice_et", 0.0) for upd in ready_updates]))

        round_metrics = {
            "train_loss": avg_train_loss,
            "val_dice_mean": avg_dice_mean,
            "val_dice_tc": avg_dice_tc,
            "val_dice_wt": avg_dice_wt,
            "val_dice_et": avg_dice_et,
            "num_clients": upload_count,
            **agg_metrics,
        }

        prev_best = self.model_manager.best_dice
        latest_ckpt_path = self.model_manager.save_round_checkpoint(
            round_num=target_round,
            model=self.model_manager.model,
            metrics=round_metrics,
            participating_clients=participating_clients,
        )
        is_new_best = self.model_manager.best_dice > prev_best

        elapsed_sec = time.time() - start_time

        # 4. Construct RoundCompleteEvent
        event = RoundCompleteEvent(
            round_num=target_round,
            participating_clients=participating_clients,
            num_samples_total=total_samples,
            metrics=round_metrics,
            checkpoint_path=str(latest_ckpt_path),
            is_new_best=is_new_best,
            duration_seconds=round(elapsed_sec, 3),
        )

        # 5. Advance Round in SyncManager
        if self.auto_advance_round:
            self.sync_manager.advance_round(target_round + 1)

        # 6. Record Event & Dispatch Notifications
        self._record_event(event)
        self._completed_events.append(event)
        self._notify_subscribers(event)

        self.logger.info(
            f"[AutoAggregate] Completed Round {target_round} in {elapsed_sec:.2f}s! "
            f"Dice Mean={avg_dice_mean:.4f} (New Best: {is_new_best}) -> Saved: {latest_ckpt_path.name}"
        )

        return True, event

    def _notify_subscribers(self, event: RoundCompleteEvent) -> None:
        """Dispatch event notifications to all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                self.logger.error(f"[AutoAggregate] Error in round complete callback: {e}")

    def _record_event(self, event: RoundCompleteEvent) -> None:
        """Persist aggregation event to JSONL audit log."""
        try:
            with open(self.audit_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            self.logger.error(f"[AutoAggregate] Failed to write event log: {e}")

    def get_aggregation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read and return list of completed round aggregation events from disk."""
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
            self.logger.error(f"[AutoAggregate] Error reading event log: {e}")

        if limit is not None:
            return events[-limit:]
        return events

    def get_trigger_status(self) -> Dict[str, Any]:
        """Return status telemetry for automated aggregation engine."""
        history = self.get_aggregation_history()
        return {
            "current_round": self.sync_manager.current_round,
            "min_upload_threshold": self.min_upload_threshold,
            "pending_uploads_count": len(self.sync_manager.get_ready_updates()),
            "total_aggregated_rounds": len(history),
            "latest_event": history[-1] if history else None,
            "subscribers_count": len(self._callbacks),
        }
