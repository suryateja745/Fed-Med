"""
Server CLI Runner for FedMed Federated Learning Coordinator.
Launches Flower server with configured aggregation strategy, client threshold settings,
round callbacks, and system inspection.
"""

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.models.unet3d import build_unet3d_from_config, count_parameters
from federation.server.fl_server import (
    create_fedavg_strategy,
    get_initial_parameters,
    start_flower_server,
)
from federation.utils.config_loader import load_config
from federation.utils.logger import setup_logger


# CLI Argument Parser

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for launching the federated coordinator server.
    """
    parser = argparse.ArgumentParser(
        description="FedMed Federated Learning Server Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        help="Host and port to bind the Flower aggregation server (e.g. 0.0.0.0:8080 or 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Number of federated aggregation rounds to execute",
    )
    parser.add_argument(
        "--min-fit-clients",
        type=int,
        default=None,
        help="Minimum number of hospital clients required for local training in each round",
    )
    parser.add_argument(
        "--min-available-clients",
        type=int,
        default=None,
        help="Minimum number of connected hospital clients before training rounds begin",
    )
    parser.add_argument(
        "--min-evaluate-clients",
        type=int,
        default=None,
        help="Minimum number of hospital clients required for validation evaluation",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="FedMedStrategy",
        choices=["FedMedStrategy", "FedAvg"],
        help="Aggregation strategy to use for federated rounds",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom fl_config.yaml configuration file",
    )
    parser.add_argument(
        "--dry-run",
        "--validate-only",
        action="store_true",
        help="Validate model architecture, strategy instantiation, and configuration without starting network listener",
    )

    return parser.parse_args(args)


# Main Server Runner Entrypoint

def main(args: Optional[List[str]] = None) -> int:
    """
    Main entrypoint for server CLI execution.
    """
    parsed_args = parse_args(args)
    logger = setup_logger(name="FedMedServerRunner")
    logger.info("Initializing FedMed Federated Coordinator Server...")

    # Step 1: Load Configuration
    config = load_config(parsed_args.config)

    # Apply CLI overrides to configuration
    fed_cfg = config.setdefault("federation", {})
    strat_cfg = config.setdefault("strategy", {})
    if parsed_args.server_address is not None:
        fed_cfg["server_address"] = parsed_args.server_address
    if parsed_args.num_rounds is not None:
        fed_cfg["num_rounds"] = parsed_args.num_rounds
    if parsed_args.min_fit_clients is not None:
        fed_cfg["min_fit_clients"] = parsed_args.min_fit_clients
    if parsed_args.min_available_clients is not None:
        fed_cfg["min_available_clients"] = parsed_args.min_available_clients
    if parsed_args.min_evaluate_clients is not None:
        fed_cfg["min_evaluate_clients"] = parsed_args.min_evaluate_clients
    if parsed_args.strategy is not None:
        strat_cfg["name"] = parsed_args.strategy

    # Step 2: System Telemetry
    server_address = fed_cfg.get("server_address", "0.0.0.0:8080")
    num_rounds = fed_cfg.get("num_rounds", 10)
    min_fit = fed_cfg.get("min_fit_clients", 2)
    min_avail = fed_cfg.get("min_available_clients", 2)
    strat_name = strat_cfg.get("name", "FedMedStrategy")

    logger.info("Server Environment & Configuration:")
    logger.info(f"  * Platform OS      : {platform.system()} {platform.release()}")
    logger.info(f"  * Python Version   : {platform.python_version()}")
    logger.info(f"  * PyTorch Version  : {torch.__version__}")
    logger.info(f"  * Server Address   : {server_address}")
    logger.info(f"  * Aggregation Strat: {strat_name}")
    logger.info(f"  * Total Rounds     : {num_rounds}")
    logger.info(f"  * Min Fit Clients  : {min_fit}")
    logger.info(f"  * Min Avail Clients: {min_avail}")

    # Step 3: Global Model Instantiation
    model = build_unet3d_from_config(config)
    param_counts = count_parameters(model)
    logger.info(
        f"Global 3D U-Net Model initialized: {param_counts['trainable_parameters']:,} trainable parameters"
    )

    # Step 4: Strategy Construction
    if "fedavg" in strat_name.lower():
        strategy = create_fedavg_strategy(
            model=model,
            config=config,
            min_fit_clients=min_fit,
            min_available_clients=min_avail,
            min_evaluate_clients=fed_cfg.get("min_evaluate_clients", 2),
        )
        logger.info("Standard FedAvg aggregation strategy successfully initialized.")
    else:
        from federation.server.strategy import create_fedmed_strategy
        strategy = create_fedmed_strategy(
            model=model,
            config=config,
            min_fit_clients=min_fit,
            min_available_clients=min_avail,
            min_evaluate_clients=fed_cfg.get("min_evaluate_clients", 2),
            weighted_by_dice=strat_cfg.get("weighted_by_samples", True),
        )
        logger.info("Custom FedMedStrategy with Dice-weighted aggregation successfully initialized.")

    # Step 5: Dry-Run Mode
    if parsed_args.dry_run:
        logger.info("Dry-run validation successful. Server setup is valid and ready to run.")
        return 0

    # Step 6: Start Server
    try:
        start_flower_server(
            server_address=server_address,
            num_rounds=num_rounds,
            strategy=strategy,
            model=model,
            config=config,
        )
        return 0
    except Exception as e:
        logger.error(f"Failed to start Flower server: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
