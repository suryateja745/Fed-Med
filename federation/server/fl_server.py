"""
Baseline Flower Federated Learning Server for FedMed.
Orchestrates federated training rounds across distributed hospital nodes using
the FedAvg aggregation strategy, initializes global 3D U-Net parameters,
and distributes training configurations.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

try:
    import flwr as fl
    from flwr.common import (
        NDArrays,
        Parameters,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server import ServerConfig, start_server
    from flwr.server.strategy import FedAvg, Strategy
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False
    # Fallback placeholders for typing and offline unit testing
    class Parameters:
        def __init__(self, tensors: List[bytes] = None, tensor_type: str = "numpy.ndarray"):
            self.tensors = tensors or []
            self.tensor_type = tensor_type

    class Strategy:
        pass

    class FedAvg(Strategy):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ServerConfig:
        def __init__(self, num_rounds: int = 1):
            self.num_rounds = num_rounds

    def ndarrays_to_parameters(ndarrays: List[np.ndarray]) -> Parameters:
        return Parameters(tensors=[arr.tobytes() for arr in ndarrays])

    def parameters_to_ndarrays(parameters: Parameters) -> List[np.ndarray]:
        return []

from federation.models.unet3d import build_unet3d_from_config, get_model_parameters
from federation.utils.config_loader import load_config
from federation.utils.logger import setup_logger


# Global Parameter Initialization

def get_initial_parameters(
    model: Optional[nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Initialize global model parameters from 3D U-Net architecture.

    Args:
        model: Optional pre-constructed PyTorch model instance.
        config: Optional configuration dictionary.

    Returns:
        Flower Parameters object (or List[np.ndarray] if flwr is unavailable).
    """
    if model is None:
        model = build_unet3d_from_config(config)

    ndarrays = get_model_parameters(model)
    return ndarrays_to_parameters(ndarrays)


# Dynamic Federated Config Callbacks

def get_fit_config_fn(config: Optional[Dict[str, Any]] = None) -> Callable[[int], Dict[str, Any]]:
    """
    Generate client training configuration callback for each federated round.
    Sends round number, local epochs, batch size, learning rate, and loss settings to clients.
    """
    if config is None:
        config = load_config()

    training_cfg = config.get("training", {})

    def fit_config(server_round: int) -> Dict[str, Any]:
        return {
            "server_round": server_round,
            "local_epochs": int(training_cfg.get("local_epochs", 3)),
            "batch_size": int(training_cfg.get("batch_size", 2)),
            "learning_rate": float(training_cfg.get("learning_rate", 2e-4)),
            "weight_decay": float(training_cfg.get("weight_decay", 1e-5)),
            "optimizer": str(training_cfg.get("optimizer", "AdamW")),
            "loss_function": str(training_cfg.get("loss_function", "DiceCELoss")),
            "gradient_clip_val": float(training_cfg.get("gradient_clip_val", 1.0)),
        }

    return fit_config


def get_evaluate_config_fn(config: Optional[Dict[str, Any]] = None) -> Callable[[int], Dict[str, Any]]:
    """
    Generate client evaluation configuration callback for each federated round.
    """
    if config is None:
        config = load_config()

    training_cfg = config.get("training", {})

    def evaluate_config(server_round: int) -> Dict[str, Any]:
        return {
            "server_round": server_round,
            "loss_function": str(training_cfg.get("loss_function", "DiceCELoss")),
        }

    return evaluate_config


# Baseline FedAvg Strategy Factory

def create_fedavg_strategy(
    model: Optional[nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
    fraction_fit: Optional[float] = None,
    fraction_evaluate: Optional[float] = None,
    min_fit_clients: Optional[int] = None,
    min_evaluate_clients: Optional[int] = None,
    min_available_clients: Optional[int] = None,
    fit_metrics_aggregation_fn: Optional[Callable] = None,
    evaluate_metrics_aggregation_fn: Optional[Callable] = None,
) -> FedAvg:
    """
    Construct standard Flower FedAvg strategy initialized with global 3D U-Net weights
    and round configuration handlers.

    Returns:
        Configured FedAvg strategy instance.
    """
    if config is None:
        config = load_config()

    fed_cfg = config.get("federation", {})

    frac_fit = fraction_fit if fraction_fit is not None else float(fed_cfg.get("fraction_fit", 1.0))
    frac_eval = fraction_evaluate if fraction_evaluate is not None else float(fed_cfg.get("fraction_evaluate", 1.0))
    m_fit = min_fit_clients if min_fit_clients is not None else int(fed_cfg.get("min_fit_clients", 2))
    m_eval = min_evaluate_clients if min_evaluate_clients is not None else int(fed_cfg.get("min_evaluate_clients", 2))
    m_avail = min_available_clients if min_available_clients is not None else int(fed_cfg.get("min_available_clients", 2))

    initial_params = get_initial_parameters(model=model, config=config)
    on_fit_fn = get_fit_config_fn(config)
    on_eval_fn = get_evaluate_config_fn(config)

    strategy = FedAvg(
        fraction_fit=frac_fit,
        fraction_evaluate=frac_eval,
        min_fit_clients=m_fit,
        min_evaluate_clients=m_eval,
        min_available_clients=m_avail,
        initial_parameters=initial_params,
        on_fit_config_fn=on_fit_fn,
        on_evaluate_config_fn=on_eval_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    return strategy


# Flower Server Launcher

def start_flower_server(
    server_address: Optional[str] = None,
    num_rounds: Optional[int] = None,
    strategy: Optional[Strategy] = None,
    model: Optional[nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Start Flower federated learning coordinator server.

    Args:
        server_address: Host and port to bind server (e.g. 0.0.0.0:8080).
        num_rounds: Number of federated training rounds to execute.
        strategy: Aggregation strategy instance (defaults to FedAvg).
        model: Model architecture instance.
        config: Configuration dictionary.
    """
    if not HAS_FLWR:
        raise ImportError("Flower (flwr) must be installed to start the server.")

    if config is None:
        config = load_config()

    fed_cfg = config.get("federation", {})
    address = server_address or fed_cfg.get("server_address", "0.0.0.0:8080")
    rounds = num_rounds if num_rounds is not None else int(fed_cfg.get("num_rounds", 10))

    if strategy is None:
        strategy = create_fedavg_strategy(model=model, config=config)

    logger = setup_logger(name="FedMedServer")
    logger.info(f"Starting FedMed Flower Server on {address} for {rounds} rounds...")

    server_config = ServerConfig(num_rounds=rounds)

    history = fl.server.start_server(
        server_address=address,
        config=server_config,
        strategy=strategy,
    )

    logger.info("Federated Learning session completed successfully.")
    return history
