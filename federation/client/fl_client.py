"""
Flower NumPyClient implementation for local hospital nodes in FedMed.
Handles parameter extraction, parameter injection, local training (fit),
and validation evaluation (evaluate) for 3D Brain Tumor MRI segmentation.
"""

from pathlib import Path
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


class FedMedClient(NumPyClient):
    """
    Flower NumPyClient worker for local hospital nodes in FedMed.

    Handles:
    - Extracting local PyTorch model weights to NumPy arrays for Flower aggregation.
    - Injecting aggregated global parameters into the local model.
    - Running local training epochs on private hospital MRI scans.
    - Computing validation loss and multi-class Dice metrics (WT, TC, ET).
    - Preserving patient privacy by never exposing raw MRI scans or patient records.
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
        and return updated parameters with training metrics.

        Args:
            parameters: Global model parameters from Flower server.
            config: Federated round configuration dictionary sent by server.

        Returns:
            Tuple of (updated_parameters, num_train_samples, metrics_dict).
        """
        # Step 1: Update local model with received global parameters
        if parameters:
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

        # Step 3: Run local training epochs
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

        # Step 4: Extract updated parameters
        updated_parameters = self.get_parameters()
        num_samples = int(
            train_results.get(
                "num_samples",
                len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else len(self.train_loader),
            )
        )

        # Step 5: Format metrics dictionary for Flower
        metrics = {
            "hospital_id": self.hospital_id,
            "train_loss": float(train_results.get("train_loss", 0.0)),
            "epochs_completed": int(train_results.get("epochs_completed", local_epochs)),
        }
        for key in ["val_loss", "val_dice_mean", "val_dice_tc", "val_dice_wt", "val_dice_et"]:
            if key in train_results:
                metrics[key] = float(train_results[key])

        self.logger.info(
            f"[{self.hospital_id}] Completed Round {server_round} training: "
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
        # Step 1: Update local model with received global parameters
        if parameters:
            self.set_parameters(parameters)

        eval_loader = self.val_loader if self.val_loader is not None and len(self.val_loader) > 0 else self.train_loader
        if eval_loader is None or len(eval_loader) == 0:
            self.logger.warning(f"[{self.hospital_id}] No evaluation data available.")
            return 0.0, 0, {"val_dice_mean": 0.0, "hospital_id": self.hospital_id}

        loss_name = str(config.get("loss_function", self.config.get("training", {}).get("loss_function", "DiceCELoss")))
        loss_fn = get_loss_function(loss_name)

        # Step 2: Execute validation loop
        val_results = validate(
            model=self.model,
            val_loader=eval_loader,
            loss_fn=loss_fn,
            device=self.device,
        )

        val_loss = float(val_results.get("val_loss", 0.0))
        num_samples = (
            len(eval_loader.dataset)
            if hasattr(eval_loader, "dataset") and eval_loader.dataset is not None
            else len(eval_loader)
        )

        metrics = {
            "hospital_id": self.hospital_id,
            "val_loss": val_loss,
            "val_dice_mean": float(val_results.get("val_dice_mean", 0.0)),
            "val_dice_tc": float(val_results.get("val_dice_tc", 0.0)),
            "val_dice_wt": float(val_results.get("val_dice_wt", 0.0)),
            "val_dice_et": float(val_results.get("val_dice_et", 0.0)),
        }

        self.logger.info(
            f"[{self.hospital_id}] Evaluation completed: "
            f"loss={val_loss:.4f}, dice_mean={metrics['val_dice_mean']:.4f}, "
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
