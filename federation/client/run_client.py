"""
Client CLI Runner for Local Hospital Nodes in FedMed.
Enables hospital nodes to launch Flower NumPyClient workers, dynamically
configure local MRI dataset directories, validate local imaging datasets,
inspect compute hardware, and connect to the federated aggregation server.
"""

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.client.fl_client import (
    FedMedClient,
    create_client,
    start_fedmed_client,
)
from federation.datasets.partitioner import (
    generate_synthetic_mri_dataset,
    scan_local_dataset,
)
from federation.utils.config_loader import load_config
from federation.utils.logger import setup_logger


# Hardware & System Telemetry Inspection

def inspect_hardware_environment(device_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspect local host hardware, OS environment, CPU cores, GPU availability,
    and PyTorch execution backend.

    Returns:
        Structured dictionary containing hardware telemetry.
    """
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"

    gpu_memory_gb = 0.0
    if cuda_available:
        try:
            gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        except Exception:
            gpu_memory_gb = 0.0

    target_device = device_override or ("cuda" if cuda_available else "cpu")

    return {
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_count": device_count,
        "gpu_name": device_name,
        "gpu_memory_gb": gpu_memory_gb,
        "cpu_count": os.cpu_count() or 1,
        "target_device": target_device,
    }


# Local Dataset Directory Validation

def validate_dataset_directory(
    data_dir: Union[str, Path],
    hospital_id: str = "hospital",
    create_synthetic_if_empty: bool = False,
    num_synthetic_samples: int = 6,
    spatial_size: Tuple[int, int, int] = (32, 32, 32),
) -> List[Dict[str, str]]:
    """
    Validate local hospital dataset directory structure.
    Verifies that the directory exists and contains valid paired MRI scans and segmentation masks.
    Optionally bootstraps synthetic 3D MRI volumes for zero-data local testing.

    Args:
        data_dir: Path to local MRI dataset folder.
        hospital_id: Hospital identifier.
        create_synthetic_if_empty: If True, generates synthetic samples if folder is empty/missing.
        num_synthetic_samples: Number of synthetic MRI volumes to generate if bootstrapping.
        spatial_size: Spatial dimensions for synthetic volumes.

    Returns:
        List of paired sample dictionaries: [{'image': path, 'label': path}, ...]

    Raises:
        FileNotFoundError: If directory does not exist and auto-bootstrap is False.
        ValueError: If directory exists but contains no valid paired MRI files.
    """
    path = Path(data_dir)

    if not path.exists():
        if create_synthetic_if_empty:
            path.mkdir(parents=True, exist_ok=True)
            return generate_synthetic_mri_dataset(
                output_dir=path,
                num_samples=num_synthetic_samples,
                spatial_size=spatial_size,
            )
        raise FileNotFoundError(f"Local dataset directory does not exist: {path}")

    # Scan dataset directory for paired files
    data_list = scan_local_dataset(path)

    if not data_list:
        if create_synthetic_if_empty:
            return generate_synthetic_mri_dataset(
                output_dir=path,
                num_samples=num_synthetic_samples,
                spatial_size=spatial_size,
            )
        raise ValueError(
            f"No valid paired MRI scans and segmentation masks found in: {path}. "
            f"Expected formats: (.nii, .nii.gz, .npy, .npz) with matching image/label naming."
        )

    # Validate that scanned files are accessible
    valid_samples = []
    for item in data_list:
        img_path = Path(item["image"])
        lbl_path = Path(item["label"])
        if img_path.exists() and lbl_path.exists():
            valid_samples.append(item)

    if not valid_samples:
        raise ValueError(f"Found {len(data_list)} catalog entries, but underlying files are missing on disk.")

    return valid_samples


# CLI Argument Parser

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for launching a hospital client node.
    """
    parser = argparse.ArgumentParser(
        description="FedMed Hospital Client CLI Runner",
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
        help="Path to local MRI dataset folder on hospital storage. If not specified, defaults to ./data/<hospital-id>",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        help="Flower aggregation server address (e.g. 127.0.0.1:8080 or domain:port)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom fl_config.yaml configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override local training epochs per federated round",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override local training batch size",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=None,
        help="Override local optimizer learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Execution device override (cpu or cuda)",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        help="Spatial patch crop size for 3D MRI preprocessing (D H W)",
    )
    parser.add_argument(
        "--create-synthetic",
        action="store_true",
        help="Automatically generate synthetic 3D MRI data if dataset directory is empty or missing",
    )
    parser.add_argument(
        "--num-synthetic-samples",
        type=int,
        default=6,
        help="Number of synthetic samples to generate when --create-synthetic is enabled",
    )
    parser.add_argument(
        "--dry-run",
        "--validate-only",
        action="store_true",
        help="Perform hardware check, dataset validation, model instantiation, and a local dry-run step without connecting to Flower server",
    )

    return parser.parse_args(args)


# Main Client Runner Entrypoint

def main(args: Optional[List[str]] = None) -> int:
    """
    Main entrypoint for hospital client CLI execution.
    """
    parsed_args = parse_args(args)
    hospital_id = parsed_args.hospital_id
    logger = setup_logger(name=f"FedMedRunner-{hospital_id}")

    logger.info(f"Launching FedMed Client CLI Runner for node: '{hospital_id}'")

    # Step 1: Load Configuration
    config = load_config(parsed_args.config)

    # Apply CLI Overrides to configuration
    if parsed_args.epochs is not None:
        config.setdefault("training", {})["local_epochs"] = parsed_args.epochs
    if parsed_args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = parsed_args.batch_size
    if parsed_args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = parsed_args.lr
    if parsed_args.server_address is not None:
        config.setdefault("federation", {})["server_address"] = parsed_args.server_address

    # Step 2: Hardware Environment Inspection
    hw_info = inspect_hardware_environment(device_override=parsed_args.device)
    logger.info("System & Compute Hardware Telemetry:")
    logger.info(f"  * Platform OS   : {hw_info['os']}")
    logger.info(f"  * Python Version: {hw_info['python_version']}")
    logger.info(f"  * PyTorch       : {hw_info['pytorch_version']}")
    logger.info(f"  * CPU Cores     : {hw_info['cpu_count']}")
    logger.info(f"  * CUDA Available: {hw_info['cuda_available']} (GPUs: {hw_info['gpu_count']}, Model: {hw_info['gpu_name']}, VRAM: {hw_info['gpu_memory_gb']} GB)")
    logger.info(f"  * Target Device : {hw_info['target_device']}")

    # Step 3: Dataset Directory Resolution & Validation
    data_dir = parsed_args.data_dir
    if data_dir is None:
        data_dir = str(Path("./data") / hospital_id)

    logger.info(f"Validating local hospital dataset path: {data_dir}")
    try:
        data_samples = validate_dataset_directory(
            data_dir=data_dir,
            hospital_id=hospital_id,
            create_synthetic_if_empty=parsed_args.create_synthetic,
            num_synthetic_samples=parsed_args.num_synthetic_samples,
            spatial_size=tuple(parsed_args.roi_size),
        )
        logger.info(f"Dataset validation successful. Found {len(data_samples)} paired MRI samples for '{hospital_id}'.")
    except Exception as e:
        logger.error(f"Dataset validation failed for '{hospital_id}': {e}")
        return 1

    # Step 4: Client Instantiation
    roi_size = tuple(parsed_args.roi_size)
    try:
        client = create_client(
            hospital_id=hospital_id,
            data_list=data_samples,
            config=config,
            batch_size=config.get("training", {}).get("batch_size", 2),
            roi_size=roi_size,
            device=hw_info["target_device"],
        )
        logger.info(f"Successfully initialized FedMedClient for '{hospital_id}'.")
    except Exception as e:
        logger.error(f"Failed to instantiate FedMedClient: {e}")
        return 1

    # Step 5: Dry-Run / Validation Mode (Optional)
    if parsed_args.dry_run:
        logger.info("Running local offline self-test (1-round dry run)...")
        initial_params = client.get_parameters()
        test_config = {
            "server_round": 1,
            "local_epochs": 1,
            "learning_rate": config.get("training", {}).get("learning_rate", 1e-3),
        }
        updated_params, num_samples, train_metrics = client.fit(initial_params, test_config)
        val_loss, num_val, val_metrics = client.evaluate(updated_params, test_config)

        logger.info(f"Dry-run self-test completed successfully:")
        logger.info(f"  * Training Loss : {train_metrics.get('train_loss', 'N/A')}")
        logger.info(f"  * Validation Loss: {val_loss:.4f}")
        logger.info(f"  * Validation Dice Mean: {val_metrics.get('val_dice_mean', 'N/A')}")
        logger.info("Client is healthy and ready for federated training rounds.")
        return 0

    # Step 6: Connect to Flower Server
    server_address = config.get("federation", {}).get("server_address", "127.0.0.1:8080")
    logger.info(f"Connecting to Flower server at {server_address}...")
    try:
        start_fedmed_client(client, server_address=server_address)
        return 0
    except Exception as e:
        logger.error(f"Error occurred during Flower federated session: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
