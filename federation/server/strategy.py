"""
Custom FedMed Federated Aggregation Strategy for 3D Brain Tumor MRI Segmentation.
Implements FedMedStrategy extending Flower's FedAvg with:
- Sample-weighted and Dice-weighted parameter aggregation.
- Aggregated multi-region Dice metrics evaluation (Mean, TC, WT, ET).
- Server-side centralized validation hook against benchmark/holdout MRI datasets.
- Detailed round metrics consolidation and logging.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import flwr as fl
    from flwr.common import (
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        NDArrays,
        Parameters,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False
    # Fallback types for offline development and testing
    class Parameters:
        def __init__(self, tensors: List[bytes] = None, tensor_type: str = "numpy.ndarray"):
            self.tensors = tensors or []
            self.tensor_type = tensor_type

    class FitRes:
        def __init__(self, status=None, parameters=None, num_examples=0, metrics=None):
            self.parameters = parameters
            self.num_examples = num_examples
            self.metrics = metrics or {}

    class EvaluateRes:
        def __init__(self, status=None, loss=0.0, num_examples=0, metrics=None):
            self.loss = loss
            self.num_examples = num_examples
            self.metrics = metrics or {}

    class ClientProxy:
        def __init__(self, cid: str = "client"):
            self.cid = cid

    class FedAvg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def ndarrays_to_parameters(ndarrays: List[np.ndarray]) -> Parameters:
        return Parameters(tensors=[arr.tobytes() for arr in ndarrays])

    def parameters_to_ndarrays(parameters: Parameters) -> List[np.ndarray]:
        return []

from federation.models.metrics import get_loss_function
from federation.models.trainer import validate
from federation.models.unet3d import build_unet3d_from_config, set_model_parameters
from federation.server.fl_server import (
    get_evaluate_config_fn,
    get_fit_config_fn,
    get_initial_parameters,
)
from federation.utils.config_loader import load_config
from federation.utils.logger import setup_logger


# Parameter Weighted Aggregation Helper

def aggregate_weighted_parameters(results: List[Tuple[List[np.ndarray], float]]) -> List[np.ndarray]:
    """
    Compute weighted element-wise average across multiple client parameter lists:
    W_bar = sum(w_i * W_i) / sum(w_i)

    Args:
        results: List of tuples containing (client_parameters, weight).

    Returns:
        List of aggregated NumPy ndarrays matching original shapes and types.
    """
    if not results:
        return []

    total_weight = sum(weight for _, weight in results)
    if total_weight <= 0.0:
        return results[0][0]

    first_params = results[0][0]
    aggregated = [
        np.zeros_like(p, dtype=np.float64 if np.issubdtype(p.dtype, np.floating) else p.dtype)
        for p in first_params
    ]

    for params, weight in results:
        normalized_w = float(weight / total_weight)
        for idx, p in enumerate(params):
            if np.issubdtype(p.dtype, np.floating):
                aggregated[idx] += p * normalized_w
            elif np.issubdtype(p.dtype, np.integer):
                aggregated[idx] = np.round(aggregated[idx] + p * normalized_w).astype(p.dtype)
            else:
                aggregated[idx] = p

    return [
        p.astype(first_params[idx].dtype) if np.issubdtype(first_params[idx].dtype, np.floating) else p
        for idx, p in enumerate(aggregated)
    ]


# Metrics Aggregation Functions

def aggregate_fit_metrics(results: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Aggregate training metrics returned from multiple hospital nodes.

    Args:
        results: List of (num_examples, metrics_dict) from client fit results.

    Returns:
        Aggregated metrics dictionary with weighted average train loss and durations.
    """
    if not results:
        return {}

    total_samples = sum(num_examples for num_examples, _ in results)
    if total_samples == 0:
        return {}

    weighted_loss = sum(m.get("train_loss", 0.0) * num_examples for num_examples, m in results)
    avg_loss = weighted_loss / total_samples

    epoch_durations = [m["epoch_duration"] for _, m in results if "epoch_duration" in m]
    round_durations = [m["round_duration"] for _, m in results if "round_duration" in m]

    aggregated = {
        "train_loss": round(float(avg_loss), 4),
        "total_fit_samples": total_samples,
        "num_clients_fit": len(results),
    }

    if epoch_durations:
        aggregated["avg_epoch_duration"] = round(float(np.mean(epoch_durations)), 4)
    if round_durations:
        aggregated["avg_round_duration"] = round(float(np.mean(round_durations)), 4)

    return aggregated


def aggregate_evaluate_metrics(results: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Aggregate evaluation Dice scores and validation loss across hospital clients.

    Args:
        results: List of (num_examples, metrics_dict) from client evaluate results.

    Returns:
        Consolidated dictionary with global Dice scores (Mean, TC, WT, ET).
    """
    if not results:
        return {}

    total_samples = sum(num_examples for num_examples, _ in results)
    if total_samples == 0:
        return {}

    metrics_keys = ["dice_score", "val_dice_mean", "val_dice_tc", "val_dice_wt", "val_dice_et", "val_loss"]
    aggregated = {
        "total_eval_samples": total_samples,
        "num_clients_evaluated": len(results),
    }

    for key in metrics_keys:
        weighted_sum = sum(
            m.get(key, 0.0) * num_examples
            for num_examples, m in results
            if key in m
        )
        samples_with_metric = sum(
            num_examples
            for num_examples, m in results
            if key in m
        )
        if samples_with_metric > 0:
            aggregated[key] = round(float(weighted_sum / samples_with_metric), 4)

    return aggregated


# Centralized Benchmark Evaluation Hook

def get_server_eval_fn(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Callable[[int, Any, Dict[str, Any]], Optional[Tuple[float, Dict[str, Any]]]]:
    """
    Construct centralized server evaluation hook for benchmark testing
    against a holdout validation dataset after each federated round.

    Returns:
        Evaluation function compatible with Flower Strategy evaluate_fn.
    """
    logger = setup_logger(name="ServerEval")
    loss_function = loss_fn or get_loss_function("DiceCELoss")

    def server_evaluate(
        server_round: int,
        parameters: Any,
        config: Dict[str, Any],
    ) -> Optional[Tuple[float, Dict[str, Any]]]:
        if parameters is None:
            return None

        # Inject global parameters into server evaluation model
        if hasattr(parameters, "tensors"):
            ndarrays = parameters_to_ndarrays(parameters)
        elif isinstance(parameters, list):
            ndarrays = parameters
        else:
            return None

        set_model_parameters(model, ndarrays)

        # Run centralized validation
        val_results = validate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_function,
            device=device,
        )

        loss = float(val_results.get("val_loss", 0.0))
        metrics = {
            "server_round": server_round,
            "val_loss": loss,
            "dice_score": float(val_results.get("val_dice_mean", 0.0)),
            "dice_mean": float(val_results.get("val_dice_mean", 0.0)),
            "dice_tc": float(val_results.get("val_dice_tc", 0.0)),
            "dice_wt": float(val_results.get("val_dice_wt", 0.0)),
            "dice_et": float(val_results.get("val_dice_et", 0.0)),
        }

        logger.info(
            f"[Server Benchmark Round {server_round}] Loss: {loss:.4f}, "
            f"Dice Mean: {metrics['dice_mean']:.4f} (TC: {metrics['dice_tc']:.4f}, "
            f"WT: {metrics['dice_wt']:.4f}, ET: {metrics['dice_et']:.4f})"
        )

        return loss, metrics

    return server_evaluate


# Custom FedMed Strategy

class FedMedStrategy(FedAvg):
    """
    Custom Flower federated learning strategy for 3D Brain Tumor MRI Segmentation.

    Key Features:
    - Weighted parameter aggregation based on client sample sizes and local Dice scores.
    - Comprehensive multi-region Dice aggregation across all hospital silos.
    - Centralized server-side evaluation against holdout datasets.
    - Round history logging and best global Dice score tracking.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        weighted_by_dice: bool = True,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
    ) -> None:
        if config is None:
            config = load_config()

        self.model = model
        self.config = config
        self.weighted_by_dice = weighted_by_dice
        self.logger = setup_logger(name="FedMedStrategy")

        self.round_history: List[Dict[str, Any]] = []
        self.best_global_dice: float = -1.0

        fed_cfg = config.get("federation", {})
        initial_params = get_initial_parameters(model=model, config=config)
        fit_cfg_fn = on_fit_config_fn or get_fit_config_fn(config)
        eval_cfg_fn = on_evaluate_config_fn or get_evaluate_config_fn(config)

        super().__init__(
            fraction_fit=fraction_fit or float(fed_cfg.get("fraction_fit", 1.0)),
            fraction_evaluate=fraction_evaluate or float(fed_cfg.get("fraction_evaluate", 1.0)),
            min_fit_clients=min_fit_clients or int(fed_cfg.get("min_fit_clients", 2)),
            min_evaluate_clients=min_evaluate_clients or int(fed_cfg.get("min_evaluate_clients", 2)),
            min_available_clients=min_available_clients or int(fed_cfg.get("min_available_clients", 2)),
            initial_parameters=initial_params,
            on_fit_config_fn=fit_cfg_fn,
            on_evaluate_config_fn=eval_cfg_fn,
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        )

        self.logger.info(
            f"Initialized FedMedStrategy (min_fit={self.min_fit_clients}, "
            f"min_avail={self.min_available_clients}, weighted_by_dice={self.weighted_by_dice})"
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        failures: List[Union[Tuple[Any, Any], BaseException]],
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        Aggregate local model updates from hospital clients using sample-count
        and validation Dice score weighting.

        Returns:
            Tuple of (aggregated_parameters, aggregated_fit_metrics).
        """
        if not results:
            self.logger.warning(f"[Round {server_round}] No client results received for aggregation.")
            return None, {}

        if failures:
            self.logger.warning(f"[Round {server_round}] Received {len(failures)} client failures.")

        # Extract parameters, sample counts, and metrics
        weights_results: List[Tuple[List[np.ndarray], float]] = []
        client_metrics_list: List[Tuple[int, Dict[str, Any]]] = []

        for client, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters) if hasattr(fit_res, "parameters") else []
            num_samples = fit_res.num_examples if hasattr(fit_res, "num_examples") else 1
            metrics = fit_res.metrics if hasattr(fit_res, "metrics") else {}

            # Calculate client aggregation weight
            if self.weighted_by_dice:
                client_dice = float(metrics.get("dice_score", metrics.get("val_dice_mean", 0.0)))
                # Quality-weighted boost: w_i = n_i * (0.5 + 0.5 * dice)
                aggregation_weight = float(num_samples * (0.5 + 0.5 * max(0.0, min(1.0, client_dice))))
            else:
                aggregation_weight = float(num_samples)

            weights_results.append((params, aggregation_weight))
            client_metrics_list.append((num_samples, metrics))

        # Perform weighted parameter averaging
        aggregated_ndarrays = aggregate_weighted_parameters(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate training metrics
        aggregated_metrics = aggregate_fit_metrics(client_metrics_list)
        aggregated_metrics["server_round"] = server_round

        self.logger.info(
            f"[Round {server_round}] Aggregated {len(results)} clients "
            f"(Total samples: {aggregated_metrics.get('total_fit_samples', 'N/A')}, "
            f"Avg train_loss: {aggregated_metrics.get('train_loss', 'N/A')})"
        )

        return parameters_aggregated, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],
        failures: List[Union[Tuple[Any, Any], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Aggregate validation evaluation results and compute global multi-region Dice metrics.

        Returns:
            Tuple of (aggregated_val_loss, aggregated_evaluate_metrics).
        """
        if not results:
            return None, {}

        # Compute sample-weighted average loss
        total_samples = sum(eval_res.num_examples for _, eval_res in results)
        if total_samples == 0:
            return None, {}

        weighted_loss = sum(eval_res.loss * eval_res.num_examples for _, eval_res in results)
        avg_loss = float(weighted_loss / total_samples)

        # Aggregate evaluation metrics
        eval_metrics_list = [(eval_res.num_examples, eval_res.metrics) for _, eval_res in results]
        aggregated_metrics = aggregate_evaluate_metrics(eval_metrics_list)
        aggregated_metrics["server_round"] = server_round
        aggregated_metrics["val_loss"] = round(avg_loss, 4)

        # Track best global Dice score
        current_dice = float(aggregated_metrics.get("val_dice_mean", aggregated_metrics.get("dice_score", 0.0)))
        if current_dice > self.best_global_dice:
            prev_best = self.best_global_dice
            self.best_global_dice = current_dice
            aggregated_metrics["is_best"] = True
            self.logger.info(
                f"[Round {server_round}] New global best Dice score: {prev_best:.4f} -> {current_dice:.4f}"
            )

        self.logger.info(
            f"[Round {server_round} Evaluation] Loss: {avg_loss:.4f}, "
            f"Dice Mean: {current_dice:.4f} "
            f"(TC: {aggregated_metrics.get('val_dice_tc', 'N/A')}, "
            f"WT: {aggregated_metrics.get('val_dice_wt', 'N/A')}, "
            f"ET: {aggregated_metrics.get('val_dice_et', 'N/A')})"
        )

        return avg_loss, aggregated_metrics


# Factory Function

def create_fedmed_strategy(
    model: Optional[nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
    weighted_by_dice: bool = True,
    val_loader: Optional[DataLoader] = None,
    device: str = "cpu",
    **kwargs: Any,
) -> FedMedStrategy:
    """
    Factory helper to instantiate a fully configured FedMedStrategy.
    Optionally sets up centralized server validation if val_loader is provided.
    """
    if config is None:
        config = load_config()

    if model is None:
        model = build_unet3d_from_config(config)

    eval_fn = None
    if val_loader is not None and len(val_loader) > 0:
        eval_fn = get_server_eval_fn(
            model=model,
            val_loader=val_loader,
            device=device,
        )

    return FedMedStrategy(
        model=model,
        config=config,
        weighted_by_dice=weighted_by_dice,
        evaluate_fn=eval_fn,
        **kwargs,
    )
