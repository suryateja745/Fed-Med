"""
Multi-Hospital Federated Simulation Testbed for FedMed.
Orchestrates fast local multi-client simulation across multiple hospital nodes (e.g. Hospital A, B, C)
with non-IID/IID partitioned 3D MRI datasets, custom FedMedStrategy aggregation,
and automated multi-round convergence visualization (round_vs_dice.png, round_vs_loss.png).
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.client.fl_client import FedMedClient, create_client
from federation.datasets.partitioner import (
    create_hospital_dataloaders,
    generate_synthetic_mri_dataset,
    partition_dataset,
)
from federation.models.unet3d import (
    build_unet3d_from_config,
    create_unet3d_model,
    get_model_parameters,
    set_model_parameters,
)
from federation.server.fl_server import get_evaluate_config_fn, get_fit_config_fn
from federation.server.model_manager import GlobalModelManager
from federation.server.strategy import FedMedStrategy, create_fedmed_strategy
from federation.utils.config_loader import load_config
from federation.utils.logger import setup_logger

try:
    import flwr as fl
    from flwr.common import (
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        NDArrays,
        Parameters,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False


# Simulation Dataset & Client Factory

def setup_simulation_environment(
    num_clients: int = 3,
    data_dir: Union[str, Path] = "./data/simulated",
    partition_type: str = "quantity_skew",
    num_samples_per_client: int = 6,
    spatial_size: Tuple[int, int, int] = (32, 32, 32),
    seed: int = 42,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    """
    Prepare partitioned local datasets and directory mapping for simulated hospitals.
    Generates synthetic 3D MRI volumes if raw scans are not already present on disk.

    Returns:
        (partitions_dict, hospital_id_map)
    """
    base_dir = Path(data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    total_samples = num_clients * num_samples_per_client
    raw_data_dir = base_dir / "raw_pool"

    # Check if raw synthetic pool already exists with sufficient data
    existing_samples = []
    if raw_data_dir.exists():
        existing_imgs = list(raw_data_dir.glob("*_image.npy"))
        for img in existing_imgs:
            lbl = img.parent / img.name.replace("_image.npy", "_label.npy")
            if lbl.exists():
                existing_samples.append({"image": str(img), "label": str(lbl)})

    if len(existing_samples) < total_samples:
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        raw_samples = generate_synthetic_mri_dataset(
            output_dir=raw_data_dir,
            num_samples=total_samples,
            spatial_size=spatial_size,
            seed=seed,
        )
    else:
        raw_samples = existing_samples[:total_samples]

    partitions = partition_dataset(
        data_list=raw_samples,
        num_clients=num_clients,
        partition_type=partition_type,
        seed=seed,
    )

    hospital_map = {str(i): name for i, name in enumerate(partitions.keys())}
    return partitions, hospital_map


def build_client_factory(
    partitions: Dict[str, List[Dict[str, Any]]],
    hospital_map: Dict[str, str],
    config: Optional[Dict[str, Any]] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    device: Optional[str] = None,
    roi_size: Tuple[int, int, int] = (32, 32, 32),
) -> Callable[[str], Any]:
    """
    Create a client factory callback (client_fn) for Flower Simulation Engine.
    Maps numeric or named client IDs (cid) to their assigned hospital dataset partition.
    """
    cfg = config or load_config()

    def client_fn(cid: str):
        # Resolve hospital name from ID
        hospital_id = hospital_map.get(str(cid), f"hospital_{cid}")
        client_data_list = partitions.get(hospital_id, [])

        if not client_data_list:
            # Fallback to first available partition if CID lookup fails
            client_data_list = next(iter(partitions.values()))

        # Build local DataLoaders
        train_loader, val_loader, _, _ = create_hospital_dataloaders(
            hospital_id=hospital_id,
            data_list=client_data_list,
            batch_size=batch_size or cfg.get("training", {}).get("batch_size", 2),
            roi_size=roi_size,
        )

        # Build local 3D U-Net
        model = build_unet3d_from_config(cfg)

        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate FedMedClient
        fedmed_client = FedMedClient(
            hospital_id=hospital_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs or cfg.get("training", {}).get("local_epochs", 1),
            learning_rate=lr or cfg.get("training", {}).get("learning_rate", 0.0002),
            loss_name=cfg.get("training", {}).get("loss_function", "DiceCELoss"),
            device=target_device,
            enable_checkpointing=True,
            checkpoint_dir=f"./checkpoints/sim/{hospital_id}",
            history_file=f"./logs/sim_{hospital_id}_history.json",
        )

        if HAS_FLWR:
            try:
                return fedmed_client.to_client()
            except Exception:
                return fedmed_client
        return fedmed_client

    return client_fn


# Simulation Runner

def run_simulation(
    num_clients: int = 3,
    num_rounds: int = 3,
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 0.0002,
    partition_type: str = "quantity_skew",
    data_dir: Union[str, Path] = "./data/simulated",
    checkpoint_dir: Union[str, Path] = "./checkpoints/sim",
    plot_dir: Union[str, Path] = "./reports",
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run multi-hospital federated learning simulation.

    Args:
        num_clients: Number of hospital client nodes to simulate.
        num_rounds: Number of federated training rounds.
        epochs: Local training epochs per client per round.
        batch_size: Local batch size for training and evaluation.
        lr: Local learning rate for AdamW optimizer.
        partition_type: Data partition scheme ('quantity_skew', 'dirichlet', 'iid').
        data_dir: Directory containing or receiving simulated datasets.
        checkpoint_dir: Directory to store global and client model checkpoints.
        plot_dir: Directory to save multi-round convergence plots.
        device: 'cpu' or 'cuda'.
        config: Loaded configuration dictionary.
        dry_run: If True, executes a single fast verification step.

    Returns:
        Structured dictionary containing simulation summary metrics and history.
    """
    logger = setup_logger(name="FedMedSimulation")
    cfg = config or load_config()

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Setting up FedMed simulation with {num_clients} hospitals on device '{target_device}'...")

    # 1. Setup Partitioned Datasets
    partitions, hospital_map = setup_simulation_environment(
        num_clients=num_clients,
        data_dir=data_dir,
        partition_type=partition_type,
        num_samples_per_client=4 if dry_run else 6,
    )
    for h_name, samples in partitions.items():
        logger.info(f"  * {h_name}: {len(samples)} MRI volume samples allocated")

    # 2. Initialize Model Manager & Global Model
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    global_model = build_unet3d_from_config(cfg)
    model_manager = GlobalModelManager(
        checkpoint_dir=ckpt_path,
        model=global_model,
        export_best=True,
    )

    # 3. Build Client Generator
    client_factory = build_client_factory(
        partitions=partitions,
        hospital_map=hospital_map,
        config=cfg,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=target_device,
    )

    # 4. Initialize Aggregation Strategy
    strategy = FedMedStrategy(
        model_manager=model_manager,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        min_evaluate_clients=num_clients,
        weighted_by_dice=True,
    )

    actual_rounds = 1 if dry_run else num_rounds
    logger.info(f"Starting {actual_rounds}-round federated simulation across {num_clients} hospitals...")

    history: Dict[str, List[float]] = {
        "rounds": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice_mean": [],
        "val_dice_tc": [],
        "val_dice_wt": [],
        "val_dice_et": [],
    }

    current_parameters = get_model_parameters(global_model)

    # Standalone In-Process Simulation Orchestrator
    for r in range(1, actual_rounds + 1):
        round_start = datetime.now(timezone.utc)
        logger.info(f"--- [Simulation] Round {r}/{actual_rounds} Starting ---")

        fit_results = []
        eval_results = []
        participating_cids = []

        fit_config = {"current_round": r, "local_epochs": epochs, "learning_rate": lr}
        eval_config = {"current_round": r}

        for cid_idx in range(num_clients):
            cid_str = str(cid_idx)
            hospital_name = hospital_map[cid_str]
            client_inst = client_factory(cid_str)

            # Fit (Local Training)
            if hasattr(client_inst, "fit"):
                params_out, num_examples, metrics = client_inst.fit(current_parameters, fit_config)
            elif hasattr(client_inst, "numpy_client"):
                params_out, num_examples, metrics = client_inst.numpy_client.fit(current_parameters, fit_config)
            else:
                params_out, num_examples, metrics = current_parameters, 1, {}

            fit_results.append((client_inst, FitRes(parameters=ndarrays_to_parameters(params_out) if HAS_FLWR else None, num_examples=num_examples, metrics=metrics), params_out))
            participating_cids.append(hospital_name)

            # Evaluate (Local Validation)
            if hasattr(client_inst, "evaluate"):
                loss, eval_examples, eval_metrics = client_inst.evaluate(params_out, eval_config)
            elif hasattr(client_inst, "numpy_client"):
                loss, eval_examples, eval_metrics = client_inst.numpy_client.evaluate(params_out, eval_config)
            else:
                loss, eval_examples, eval_metrics = 1.0, 1, {}

            eval_results.append((client_inst, EvaluateRes(loss=loss, num_examples=eval_examples, metrics=eval_metrics)))

        # Aggregate Parameters
        # Strategy aggregation
        if HAS_FLWR:
            fit_tuples = [(client, fit_res) for client, fit_res, _ in fit_results]
            aggregated_params, agg_fit_metrics = strategy.aggregate_fit(server_round=r, results=fit_tuples, failures=[])
            if aggregated_params is not None:
                current_parameters = parameters_to_ndarrays(aggregated_params)
        else:
            # Native weighted aggregation
            from federation.server.strategy import aggregate_weighted_parameters
            weighted_inputs = []
            for _, fit_res, p_arr in fit_results:
                w = float(fit_res.num_examples) * (1.0 + float(fit_res.metrics.get("val_dice_mean", 0.0)))
                weighted_inputs.append((p_arr, max(0.01, w)))
            current_parameters = aggregate_weighted_parameters(weighted_inputs)

        # Update global model
        set_model_parameters(global_model, current_parameters)

        # Aggregate Evaluation Metrics
        eval_tuples = [(client, eval_res) for client, eval_res in eval_results]
        agg_loss, agg_eval_metrics = strategy.aggregate_evaluate(server_round=r, results=eval_tuples, failures=[])

        # Extract round metrics
        t_loss = float(np.mean([res.metrics.get("train_loss", 1.0) for _, res, _ in fit_results]))
        v_loss = agg_loss if agg_loss is not None else float(np.mean([res.loss for _, res in eval_results]))
        v_dice = float(agg_eval_metrics.get("val_dice_mean", 0.0))
        v_tc = float(agg_eval_metrics.get("val_dice_tc", 0.0))
        v_wt = float(agg_eval_metrics.get("val_dice_wt", 0.0))
        v_et = float(agg_eval_metrics.get("val_dice_et", 0.0))

        # Save checkpoint & update metadata
        round_metrics_dict = {
            "train_loss": t_loss,
            "val_loss": v_loss,
            "val_dice_mean": v_dice,
            "val_dice_tc": v_tc,
            "val_dice_wt": v_wt,
            "val_dice_et": v_et,
        }
        model_manager.save_round_checkpoint(
            round_num=r,
            model=global_model,
            metrics=round_metrics_dict,
            participating_clients=participating_cids,
        )

        history["rounds"].append(r)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_dice_mean"].append(v_dice)
        history["val_dice_tc"].append(v_tc)
        history["val_dice_wt"].append(v_wt)
        history["val_dice_et"].append(v_et)

        logger.info(
            f"--- [Simulation] Round {r}/{actual_rounds} Completed: "
            f"Train Loss={t_loss:.4f} | Val Loss={v_loss:.4f} | Dice Mean={v_dice:.4f} "
            f"(TC={v_tc:.4f}, WT={v_wt:.4f}, ET={v_et:.4f}) ---"
        )

    # 5. Generate and Save Convergence Plots
    plot_paths = generate_simulation_plots(history=history, output_dir=plot_dir)
    for name, p_path in plot_paths.items():
        logger.info(f"Generated convergence plot: {p_path}")

    # 6. Save Simulation Summary JSON
    summary_path = Path(plot_dir) / "simulation_summary.json"
    summary_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_clients": num_clients,
        "num_rounds": actual_rounds,
        "partition_type": partition_type,
        "best_dice": model_manager.best_dice,
        "best_round": model_manager.best_round,
        "final_metrics": {
            "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "val_dice_mean": history["val_dice_mean"][-1] if history["val_dice_mean"] else None,
        },
        "history": history,
        "plots": {k: str(v) for k, v in plot_paths.items()},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Simulation completed successfully! Summary exported to: {summary_path}")
    return summary_data


# Convergence Plot Generator

def generate_simulation_plots(
    history: Dict[str, List[float]],
    output_dir: Union[str, Path] = "./reports",
) -> Dict[str, Path]:
    """
    Generate clean, publication-ready multi-round convergence curves:
    - round_vs_dice.png: Progression of Multi-Region Dice Scores (Mean, TC, WT, ET).
    - round_vs_loss.png: Training Loss vs Validation Loss convergence.

    Args:
        history: Dictionary containing historical rounds and metric lists.
        output_dir: Destination folder for output PNG images.

    Returns:
        Dictionary mapping plot identifiers to their saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds = history.get("rounds", [])
    if not rounds:
        rounds = list(range(1, len(history.get("val_dice_mean", [])) + 1))

    plot_paths: Dict[str, Path] = {}

    # Plot 1: Round vs Multi-Region Dice Score
    dice_plot_path = out_dir / "round_vs_dice.png"
    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    if "val_dice_mean" in history and history["val_dice_mean"]:
        plt.plot(rounds, history["val_dice_mean"], marker="o", linewidth=2.5, label="Mean Dice", color="#1f77b4")
    if "val_dice_wt" in history and history["val_dice_wt"]:
        plt.plot(rounds, history["val_dice_wt"], marker="s", linestyle="--", linewidth=1.8, label="Whole Tumor (WT)", color="#2ca02c")
    if "val_dice_tc" in history and history["val_dice_tc"]:
        plt.plot(rounds, history["val_dice_tc"], marker="^", linestyle="--", linewidth=1.8, label="Tumor Core (TC)", color="#ff7f0e")
    if "val_dice_et" in history and history["val_dice_et"]:
        plt.plot(rounds, history["val_dice_et"], marker="d", linestyle=":", linewidth=1.8, label="Enhancing Tumor (ET)", color="#d62728")

    plt.title("FedMed Federated Learning: Global Validation Dice Convergence", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Federated Round Number", fontsize=11, labelpad=8)
    plt.ylabel("Dice Similarity Coefficient", fontsize=11, labelpad=8)
    plt.ylim(0.0, 1.05)
    plt.xticks(rounds)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(dice_plot_path, dpi=150)
    plt.close()
    plot_paths["round_vs_dice"] = dice_plot_path

    # Plot 2: Round vs Loss
    loss_plot_path = out_dir / "round_vs_loss.png"
    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    if "train_loss" in history and history["train_loss"]:
        plt.plot(rounds, history["train_loss"], marker="o", linewidth=2.2, label="Training Loss (Client Avg)", color="#e377c2")
    if "val_loss" in history and history["val_loss"]:
        plt.plot(rounds, history["val_loss"], marker="s", linewidth=2.2, label="Validation Loss (Global)", color="#17becf")

    plt.title("FedMed Federated Learning: Loss Convergence", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Federated Round Number", fontsize=11, labelpad=8)
    plt.ylabel("Loss (Dice + Cross Entropy)", fontsize=11, labelpad=8)
    plt.xticks(rounds)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    plot_paths["round_vs_loss"] = loss_plot_path

    return plot_paths


# CLI Entry Point

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedMed Multi-Hospital Federated Simulation Testbed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-clients", type=int, default=3, help="Number of hospital clients to simulate (e.g. 3 for Hospital A, B, C)")
    parser.add_argument("--num-rounds", type=int, default=3, help="Number of federated aggregation rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Local training epochs per client round")
    parser.add_argument("--batch-size", type=int, default=2, help="Local batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Local learning rate for AdamW")
    parser.add_argument("--partition-type", type=str, default="quantity_skew", choices=["quantity_skew", "dirichlet", "iid"], help="Dataset distribution skew among hospitals")
    parser.add_argument("--data-dir", type=str, default="./data/simulated", help="Directory for simulated datasets")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/sim", help="Directory for global and client checkpoints")
    parser.add_argument("--plot-dir", type=str, default="./reports", help="Directory to save convergence plots and summary JSON")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Execution device override (cpu/cuda)")
    parser.add_argument("--dry-run", action="store_true", help="Execute single fast validation round")

    return parser.parse_args(args)


def main():
    args = parse_args()
    run_simulation(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        partition_type=args.partition_type,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        plot_dir=args.plot_dir,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
