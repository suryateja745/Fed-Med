"""
Flower NumPyClient implementation for local hospital nodes in FedMed.
Handles parameter extraction, parameter injection, local training (fit),
validation evaluation (evaluate), checkpoint management, fallback recovery,
parameter diffing, and persistent client telemetry JSON logging.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from flwr.client import NumPyClient
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False
    class NumPyClient:
        """Fallback NumPyClient base class if Flower is not installed."""
        pass

from federation.client.checkpoint import (
    ClientCheckpointManager,
    apply_parameter_delta,
    compute_delta_statistics,
    compute_parameter_delta,
)
from federation.datasets.partitioner import create_hospital_dataloaders
from federation.models.metrics import get_loss_function
from federation.models.trainer import train_local_client, validate
from federation.models.unet3d import (
    build_unet3d_from_config,
    get_model_parameters,
    set_model_parameters,
)
from federation.utils.config_loader import load_config
from federation.utils.logger import setup_logger


# Client Telemetry History Logger

class ClientHistoryLogger:
    """
    Local client-side JSON logger for federated training rounds and evaluations.
    Maintains persistent telemetry records for auditing, performance tracking, and debugging.
    """

    def __init__(self, hospital_id: str, log_dir: Union[str, Path] = "./logs") -> None:
        self.hospital_id = str(hospital_id)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.log_dir / f"{self.hospital_id}_history.json"
        self._init_history_file()

    def _init_history_file(self) -> None:
        if not self.history_file.exists():
            initial_data = {
                "hospital_id": self.hospital_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "fit_history": [],
                "evaluate_history": [],
            }
            self._write_json(initial_data)

    def _read_json(self) -> Dict[str, Any]:
        try:
            if self.history_file.exists():
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            "hospital_id": self.hospital_id,
            "fit_history": [],
            "evaluate_history": [],
        }

    def _write_json(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def log_fit_round(self, round_data: Dict[str, Any]) -> None:
        """Append training round telemetry entry to history JSON."""
        data = self._read_json()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **round_data,
        }
        data.setdefault("fit_history", []).append(record)
        self._write_json(data)

    def log_evaluate_round(self, eval_data: Dict[str, Any]) -> None:
        """Append evaluation telemetry entry to history JSON."""
        data = self._read_json()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **eval_data,
        }
        data.setdefault("evaluate_history", []).append(record)
        self._write_json(data)

    def get_history(self) -> Dict[str, Any]:
        """Read and return full telemetry history dictionary."""
        return self._read_json()


# Core FedMed Flower Client Worker

class FedMedClient(NumPyClient):
    """
    Flower NumPyClient worker for local hospital nodes in FedMed.

    Handles:
    - Extracting local PyTorch model weights to NumPy arrays for Flower aggregation.
    - Injecting aggregated global parameters into the local model.
    - Running local training epochs on private hospital MRI scans.
    - Computing validation loss and multi-class Dice metrics (WT, TC, ET).
    - Preserving patient privacy by never exposing raw MRI scans or patient records.
    - Checkpointing local model weights and fallback recovery if local training fails.
    - Computing parameter deltas (Delta W = W_local - W_global).
    - Recording rich training telemetry and updating client_history.json.
    """

    def __init__(
        self,
        hospital_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.hospital_id = str(hospital_id)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}

        # Configure computation device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        self.model.to(self.device)
        self.logger = setup_logger(name=f"FedMedClient-{self.hospital_id}")

        # Initialize persistent JSON telemetry logger
        log_dir = self.config.get("storage", {}).get("logs_dir", "./logs")
        self.history_logger = ClientHistoryLogger(hospital_id=self.hospital_id, log_dir=log_dir)

        # Initialize checkpoint manager
        checkpoint_dir = self.config.get("storage", {}).get("checkpoint_dir", "./checkpoints")
        self.checkpoint_manager = ClientCheckpointManager(
            hospital_id=self.hospital_id,
            checkpoint_dir=checkpoint_dir,
            keep_last_n=3,
        )

        self.logger.info(
            f"Initialized FedMedClient for '{self.hospital_id}' on device '{self.device}' "
            f"(Train batches: {len(self.train_loader)}, "
            f"Val batches: {len(self.val_loader) if self.val_loader else 0})"
        )

    def get_parameters(self, config: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """
        Extract local PyTorch model state dict as a list of NumPy ndarrays.
        Used by the Flower server for aggregation (FedAvg, FedProx, etc.).
        """
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Load a list of NumPy ndarrays back into local PyTorch model state dict.
        """
        set_model_parameters(self.model, parameters)

    def get_properties(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Return client metadata properties to the Flower server.
        """
        num_train_samples = (
            len(self.train_loader.dataset)
            if hasattr(self.train_loader, "dataset") and self.train_loader.dataset is not None
            else len(self.train_loader)
        )
        num_val_samples = (
            len(self.val_loader.dataset)
            if self.val_loader is not None and hasattr(self.val_loader, "dataset") and self.val_loader.dataset is not None
            else (len(self.val_loader) if self.val_loader else 0)
        )
        return {
            "hospital_id": self.hospital_id,
            "device": self.device,
            "num_train_samples": int(num_train_samples),
            "num_val_samples": int(num_val_samples),
        }

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Receive global server parameters, train locally for E epochs on private MRI scans,
        and return updated parameters with rich training telemetry and delta statistics.

        Includes automatic fallback recovery if local training fails.

        Args:
            parameters: Global model parameters from Flower server.
            config: Federated round configuration dictionary sent by server.

        Returns:
            Tuple of (updated_parameters, num_train_samples, metrics_dict).
        """
        start_fit_time = time.perf_counter()

        # Step 1: Backup and inject global parameters
        if parameters:
            self.checkpoint_manager.backup_global_parameters(parameters)
            self.set_parameters(parameters)

        # Step 2: Parse training hyperparameters from server config or client defaults
        training_cfg = self.config.get("training", {})
        server_round = config.get("server_round", config.get("round", 1))
        local_epochs = int(config.get("local_epochs", config.get("epochs", training_cfg.get("local_epochs", 3))))
        learning_rate = float(config.get("learning_rate", config.get("lr", training_cfg.get("learning_rate", 2e-4))))
        weight_decay = float(config.get("weight_decay", training_cfg.get("weight_decay", 1e-5)))
        optimizer_name = str(config.get("optimizer", training_cfg.get("optimizer", "AdamW")))
        scheduler_name = str(config.get("scheduler", "cosine"))
        loss_name = str(config.get("loss_function", config.get("loss_name", training_cfg.get("loss_function", "DiceCELoss"))))
        gradient_clip_val = float(config.get("gradient_clip_val", training_cfg.get("gradient_clip_val", 1.0)))

        self.logger.info(
            f"[{self.hospital_id}] Starting Round {server_round} local training: "
            f"{local_epochs} epochs, lr={learning_rate}, loss={loss_name}, opt={optimizer_name}"
        )

        # Step 3: Run local training epochs with robust fallback recovery
        try:
            train_results = train_local_client(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                epochs=local_epochs,
                lr=learning_rate,
                weight_decay=weight_decay,
                optimizer_name=optimizer_name,
                scheduler_name=scheduler_name,
                loss_name=loss_name,
                device=self.device,
                gradient_clip_val=gradient_clip_val,
            )
        except Exception as e:
            self.logger.error(
                f"[{self.hospital_id}] Local training failed during Round {server_round}: {e}. "
                f"Initiating fallback recovery to global parameter baseline."
            )
            # Fallback: restore model to last global parameters
            self.checkpoint_manager.restore_global_parameters(self.model)
            fallback_params = self.get_parameters()
            error_metrics = {
                "hospital_id": self.hospital_id,
                "train_loss": float("nan"),
                "status": "failed",
                "error": str(e),
                "device": str(self.device),
            }
            return fallback_params, 0, error_metrics

        fit_duration = time.perf_counter() - start_fit_time

        # Step 4: Extract updated parameters and compute parameter deltas
        updated_parameters = self.get_parameters()
        num_samples = int(
            train_results.get(
                "num_samples",
                len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else len(self.train_loader),
            )
        )

        # Compute parameter deltas (Delta W = W_local - W_global)
        delta_stats = {}
        if parameters:
            try:
                deltas = compute_parameter_delta(updated_parameters, parameters)
                delta_stats = compute_delta_statistics(deltas)
            except Exception as e:
                self.logger.debug(f"Could not compute delta statistics: {e}")

        # Step 5: Format metrics dictionary for Flower server
        metrics = {
            "hospital_id": self.hospital_id,
            "train_loss": float(train_results.get("train_loss", 0.0)),
            "epochs_completed": int(train_results.get("epochs_completed", local_epochs)),
            "epoch_duration": float(train_results.get("epoch_duration", 0.0)),
            "round_duration": round(float(fit_duration), 4),
            "learning_rate": float(learning_rate),
            "num_samples": num_samples,
            "device": str(self.device),
        }
        if "val_loss" in train_results:
            metrics["val_loss"] = float(train_results["val_loss"])
        if "val_dice_mean" in train_results:
            metrics["dice_score"] = float(train_results["val_dice_mean"])
            metrics["val_dice_mean"] = float(train_results["val_dice_mean"])
        if "val_dice_tc" in train_results:
            metrics["val_dice_tc"] = float(train_results["val_dice_tc"])
        if "val_dice_wt" in train_results:
            metrics["val_dice_wt"] = float(train_results["val_dice_wt"])
        if "val_dice_et" in train_results:
            metrics["val_dice_et"] = float(train_results["val_dice_et"])

        # Include delta statistics in telemetry
        if delta_stats:
            metrics["delta_l2_norm"] = delta_stats.get("l2_norm", 0.0)
            metrics["delta_max_abs"] = delta_stats.get("max_abs_update", 0.0)
            metrics["delta_sparsity"] = delta_stats.get("sparsity_ratio", 0.0)

        # Step 6: Save local checkpoint and check for historical best
        self.checkpoint_manager.save_latest(
            model=self.model,
            round_num=server_round,
            metrics=metrics,
        )
        if "dice_score" in metrics:
            self.checkpoint_manager.save_best(
                model=self.model,
                round_num=server_round,
                dice_score=metrics["dice_score"],
                metrics=metrics,
            )

        # Step 7: Log telemetry to local client history JSON file
        self.history_logger.log_fit_round({
            "server_round": server_round,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "loss_function": loss_name,
            "train_loss": metrics["train_loss"],
            "epoch_losses": train_results.get("epoch_losses", []),
            "epoch_duration_seconds": metrics["epoch_duration"],
            "round_duration_seconds": metrics["round_duration"],
            "num_train_samples": num_samples,
            "val_loss": metrics.get("val_loss"),
            "dice_score": metrics.get("dice_score"),
            "val_dice_mean": metrics.get("val_dice_mean"),
            "val_dice_tc": metrics.get("val_dice_tc"),
            "val_dice_wt": metrics.get("val_dice_wt"),
            "val_dice_et": metrics.get("val_dice_et"),
            "delta_statistics": delta_stats,
            "device": self.device,
        })

        self.logger.info(
            f"[{self.hospital_id}] Completed Round {server_round} training in {fit_duration:.2f}s: "
            f"train_loss={metrics['train_loss']:.4f}, samples={num_samples}"
        )

        return updated_parameters, num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate global server parameters on local validation dataset.

        Args:
            parameters: Model parameters sent by server to evaluate.
            config: Evaluation configuration dictionary.

        Returns:
            Tuple of (val_loss, num_val_samples, metrics_dict).
        """
        start_eval_time = time.perf_counter()

        # Step 1: Update local model with received global parameters
        if parameters:
            self.set_parameters(parameters)

        eval_loader = self.val_loader if self.val_loader is not None and len(self.val_loader) > 0 else self.train_loader
        if eval_loader is None or len(eval_loader) == 0:
            self.logger.warning(f"[{self.hospital_id}] No evaluation data available.")
            return 0.0, 0, {"val_dice_mean": 0.0, "dice_score": 0.0, "hospital_id": self.hospital_id}

        loss_name = str(config.get("loss_function", self.config.get("training", {}).get("loss_function", "DiceCELoss")))
        loss_fn = get_loss_function(loss_name)

        # Step 2: Execute validation loop
        val_results = validate(
            model=self.model,
            val_loader=eval_loader,
            loss_fn=loss_fn,
            device=self.device,
        )

        eval_duration = time.perf_counter() - start_eval_time
        val_loss = float(val_results.get("val_loss", 0.0))
        dice_score = float(val_results.get("val_dice_mean", 0.0))
        num_samples = (
            len(eval_loader.dataset)
            if hasattr(eval_loader, "dataset") and eval_loader.dataset is not None
            else len(eval_loader)
        )

        metrics = {
            "hospital_id": self.hospital_id,
            "val_loss": val_loss,
            "dice_score": dice_score,
            "val_dice_mean": dice_score,
            "val_dice_tc": float(val_results.get("val_dice_tc", 0.0)),
            "val_dice_wt": float(val_results.get("val_dice_wt", 0.0)),
            "val_dice_et": float(val_results.get("val_dice_et", 0.0)),
            "eval_duration": round(float(eval_duration), 4),
            "num_samples": int(num_samples),
            "device": str(self.device),
        }

        # Step 3: Save best checkpoint if evaluated score improves
        server_round = config.get("server_round", config.get("round", 0))
        if dice_score > 0.0:
            self.checkpoint_manager.save_best(
                model=self.model,
                round_num=server_round,
                dice_score=dice_score,
                metrics=metrics,
            )

        # Step 4: Log evaluation telemetry to local client history JSON file
        self.history_logger.log_evaluate_round({
            "server_round": server_round,
            "val_loss": val_loss,
            "dice_score": dice_score,
            "val_dice_mean": dice_score,
            "val_dice_tc": metrics["val_dice_tc"],
            "val_dice_wt": metrics["val_dice_wt"],
            "val_dice_et": metrics["val_dice_et"],
            "eval_duration_seconds": metrics["eval_duration"],
            "num_val_samples": int(num_samples),
            "device": self.device,
        })

        self.logger.info(
            f"[{self.hospital_id}] Evaluation completed in {eval_duration:.2f}s: "
            f"loss={val_loss:.4f}, dice_mean={dice_score:.4f}, "
            f"dice_tc={metrics['val_dice_tc']:.4f}, dice_wt={metrics['val_dice_wt']:.4f}, "
            f"dice_et={metrics['val_dice_et']:.4f}"
        )

        return val_loss, int(num_samples), metrics


# Client Creation & Launcher Helpers

def create_client(
    hospital_id: str,
    data_dir: Optional[Union[str, Path]] = None,
    data_list: Optional[List[Dict[str, Any]]] = None,
    model: Optional[nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    device: Optional[str] = None,
) -> FedMedClient:
    """
    Factory helper to instantiate a FedMedClient with automatic DataLoader and Model setup.
    """
    if config is None:
        config = load_config()

    if model is None:
        model = build_unet3d_from_config(config)

    bs = batch_size if batch_size is not None else config.get("training", {}).get("batch_size", 2)

    if data_list is not None or data_dir is not None:
        train_loader, val_loader, _, _ = create_hospital_dataloaders(
            hospital_id=hospital_id,
            data_dir=data_dir,
            data_list=data_list,
            batch_size=bs,
            roi_size=roi_size,
        )
    else:
        raise ValueError(f"Either 'data_dir' or 'data_list' must be supplied for hospital '{hospital_id}'")

    return FedMedClient(
        hospital_id=hospital_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )


def start_fedmed_client(
    client: FedMedClient,
    server_address: Optional[str] = None,
) -> None:
    """
    Connect FedMedClient to Flower server via gRPC.
    """
    if not HAS_FLWR:
        raise ImportError("Flower (flwr) must be installed to start a gRPC client connection.")

    import flwr as fl
    addr = server_address or client.config.get("federation", {}).get("server_address", "127.0.0.1:8080")
    client.logger.info(f"Connecting {client.hospital_id} to Flower server at {addr}...")
    fl.client.start_numpy_client(server_address=addr, client=client)
