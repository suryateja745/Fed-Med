"""
FedMed Local Hospital Worker - Unified Client Entry Point.
Connects a local hospital node to the central coordinator server,
trains the 3D U-Net on private volumetric MRI scans, and logs local telemetry.
"""

import argparse
from pathlib import Path
import sys
from typing import Optional

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.client.run_client import main as run_client_main, parse_args as client_parse_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedMed Hospital Client Worker Launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hospital-id",
        type=str,
        default="hospital_a",
        help="Unique identifier for the hospital node (e.g. hospital_a, hospital_b, hospital_c)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to local MRI dataset folder. Defaults to ./data/<hospital-id>",
    )
    parser.add_argument(
        "--server-address",
        "--server",
        type=str,
        default="127.0.0.1:8080",
        help="Flower coordinator server address (e.g. 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Local training epochs per federated round",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Local batch size for training",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=None,
        help="Local learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--create-synthetic",
        action="store_true",
        default=True,
        help="Automatically generate synthetic 3D MRI scans if local folder is empty",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform local dataset validation and 1-batch training self-test without connecting to server",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Pass forwarded arguments to client runner
    raw_args = [
        "--hospital-id", args.hospital_id,
        "--server-address", args.server_address,
    ]
    if args.data_dir:
        raw_args.extend(["--data-dir", args.data_dir])
    if args.epochs is not None:
        raw_args.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        raw_args.extend(["--batch-size", str(args.batch_size)])
    if args.lr is not None:
        raw_args.extend(["--lr", str(args.lr)])
    if args.create_synthetic:
        raw_args.append("--create-synthetic")
    if args.dry_run:
        raw_args.append("--dry-run")

    run_client_main(raw_args)


if __name__ == "__main__":
    main()
