"""
FedMed Central Coordinator Server - Unified Entry Point.
Launches Flower server with FedMedStrategy, AutoDispatchTrigger, AutoAggregateTrigger,
and automatic live JSON telemetry exports for Frontend / Backend integration.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.api_bridge import FedMedAPIBridge
from federation.models.unet3d import build_unet3d_from_config, count_parameters
from federation.server.fl_server import (
    create_fedavg_strategy,
    get_evaluate_config_fn,
    get_fit_config_fn,
    get_initial_parameters,
    start_flower_server,
)
from federation.server.model_manager import GlobalModelManager
from federation.server.strategy import FedMedStrategy, create_fedmed_strategy
from federation.server.sync_manager import RoundSyncManager
from federation.server.triggers import AutoAggregateTrigger, AutoDispatchTrigger
from federation.utils.config_loader import load_config
from federation.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedMed Central Coordinator Server Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP to bind the server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen for Flower client connections")
    parser.add_argument("--rounds", type=int, default=None, help="Total federated training rounds (overrides config)")
    parser.add_argument("--min-clients", type=int, default=None, help="Minimum hospital clients required per round")
    parser.add_argument("--strategy", type=str, default="FedMedStrategy", choices=["FedMedStrategy", "FedAvg"], help="Aggregation strategy")
    parser.add_argument("--config", type=str, default=None, help="Path to custom fl_config.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Path to store model checkpoints")
    parser.add_argument("--logs-dir", type=str, default="./logs", help="Path to store log files and JSON feeds")
    parser.add_argument("--dry-run", action="store_true", help="Execute server initialization self-test and exit")
    parser.add_argument("--export-dashboard", action="store_true", default=True, help="Periodically export live dashboard JSON")

    return parser.parse_args()


def run_dashboard_exporter(
    bridge: FedMedAPIBridge,
    stop_event: threading.Event,
    interval_seconds: float = 3.0,
) -> None:
    """Background thread to continuously export live dashboard state for UI polling."""
    logger = setup_logger(name="DashboardExporter")
    logger.info("[DashboardExporter] Background JSON feed daemon started.")
    while not stop_event.is_set():
        try:
            bridge.export_dashboard_json()
        except Exception as e:
            logger.error(f"[DashboardExporter] Export error: {e}")
        time.sleep(interval_seconds)


def main() -> None:
    args = parse_args()
    logger = setup_logger(name="FedMedServer")

    config = load_config(args.config) if args.config else load_config()

    server_address = f"{args.host}:{args.port}"
    num_rounds = args.rounds if args.rounds is not None else int(config.get("federation", {}).get("num_rounds", 10))
    min_fit_clients = args.min_clients if args.min_clients is not None else int(config.get("federation", {}).get("min_fit_clients", 2))
    min_available_clients = min_fit_clients

    logger.info("Initializing FedMed Coordinator Server...")
    logger.info(f"  * Address         : {server_address}")
    logger.info(f"  * Strategy        : {args.strategy}")
    logger.info(f"  * Target Rounds   : {num_rounds}")
    logger.info(f"  * Min Fit Clients : {min_fit_clients}")
    logger.info(f"  * Checkpoints Dir : {args.checkpoint_dir}")
    logger.info(f"  * Logs & Feeds Dir: {args.logs_dir}")

    # 1. Initialize Global Model & Model Manager
    model = build_unet3d_from_config(config)
    param_stats = count_parameters(model)
    logger.info(f"  * 3D U-Net Model  : {param_stats['trainable_parameters']:,} trainable parameters")

    model_manager = GlobalModelManager(
        checkpoint_dir=args.checkpoint_dir,
        model=model,
        config=config,
        export_best=True,
    )

    # 2. Initialize API Bridge & Export Initial Dashboard State
    api_bridge = FedMedAPIBridge(
        checkpoint_dir=args.checkpoint_dir,
        logs_dir=args.logs_dir,
        config=config,
    )
    api_bridge.export_dashboard_json()

    # 3. Initialize Automated Triggers
    sync_manager = RoundSyncManager(
        initial_round=1,
        min_clients_per_round=min_fit_clients,
    )
    dispatch_trigger = AutoDispatchTrigger(
        model_manager=model_manager,
        checkpoint_dir=args.checkpoint_dir,
        audit_log_file=Path(args.logs_dir) / "dispatch_audit.jsonl",
    )

    if args.strategy == "FedMedStrategy":
        strategy = FedMedStrategy(
            model=model,
            config=config,
            model_manager=model_manager,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            weighted_by_dice=True,
        )
    else:
        strategy = create_fedavg_strategy(
            model=model,
            config=config,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
        )

    agg_trigger = AutoAggregateTrigger(
        sync_manager=sync_manager,
        strategy=strategy if isinstance(strategy, FedMedStrategy) else None,
        model_manager=model_manager,
        min_upload_threshold=min_fit_clients,
        audit_log_file=Path(args.logs_dir) / "aggregation_events.jsonl",
    )

    # Hook to re-export dashboard on round completion
    agg_trigger.register_on_round_complete(lambda evt: api_bridge.export_dashboard_json())

    if args.dry_run:
        logger.info("[Dry Run] Validating server setup and API exports...")
        state = api_bridge.get_full_dashboard_state()
        logger.info(f"[Dry Run] Successfully generated dashboard state with {len(state['system_health'])} health metrics.")
        logger.info("[Dry Run] Server coordinator initialization successful.")
        return

    # 4. Start Background Dashboard Exporter Daemon
    stop_event = threading.Event()
    if args.export_dashboard:
        exporter_thread = threading.Thread(
            target=run_dashboard_exporter,
            args=(api_bridge, stop_event, 3.0),
            daemon=True,
        )
        exporter_thread.start()

    # 5. Launch Flower Server
    try:
        start_flower_server(
            server_address=server_address,
            strategy=strategy,
            num_rounds=num_rounds,
        )
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping server...")
    finally:
        stop_event.set()
        api_bridge.export_dashboard_json()
        logger.info("FedMed Server stopped cleanly.")


if __name__ == "__main__":
    main()
