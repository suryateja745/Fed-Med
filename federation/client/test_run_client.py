"""
Unit tests for FedMed Client CLI Runner and Dataset Directory Validation.
Tests CLI argument parsing, hardware inspection telemetry, dataset directory
validation, synthetic data auto-bootstrap, and dry-run execution.
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.client.run_client import (
    inspect_hardware_environment,
    main,
    parse_args,
    validate_dataset_directory,
)
from federation.datasets.partitioner import generate_synthetic_mri_dataset


class TestClientRunner(unittest.TestCase):
    """Test suite for client CLI runner, telemetry, and dataset validator."""

    def test_argument_parsing_defaults(self):
        """Verify default CLI arguments."""
        args = parse_args([])
        self.assertEqual(args.hospital_id, "hospital_a")
        self.assertIsNone(args.data_dir)
        self.assertIsNone(args.epochs)
        self.assertIsNone(args.batch_size)
        self.assertFalse(args.dry_run)
        self.assertFalse(args.create_synthetic)

    def test_argument_parsing_custom_values(self):
        """Verify custom CLI flags."""
        args = parse_args([
            "--hospital-id", "hospital_b",
            "--data-dir", "./test_data",
            "--server-address", "192.168.1.100:8080",
            "--epochs", "5",
            "--batch-size", "4",
            "--lr", "1e-4",
            "--device", "cpu",
            "--roi-size", "32", "32", "32",
            "--create-synthetic",
            "--dry-run",
        ])
        self.assertEqual(args.hospital_id, "hospital_b")
        self.assertEqual(args.data_dir, "./test_data")
        self.assertEqual(args.server_address, "192.168.1.100:8080")
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.roi_size, [32, 32, 32])
        self.assertTrue(args.create_synthetic)
        self.assertTrue(args.dry_run)

    def test_inspect_hardware_environment(self):
        """Verify hardware inspection returns expected metadata keys."""
        hw = inspect_hardware_environment(device_override="cpu")
        self.assertIn("os", hw)
        self.assertIn("python_version", hw)
        self.assertIn("pytorch_version", hw)
        self.assertIn("cuda_available", hw)
        self.assertIn("cpu_count", hw)
        self.assertEqual(hw["target_device"], "cpu")

    def test_validate_dataset_directory_valid(self):
        """Verify directory validation on valid synthetic data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            generate_synthetic_mri_dataset(
                output_dir=tmp_dir,
                num_samples=4,
                spatial_size=(16, 16, 16),
            )
            samples = validate_dataset_directory(tmp_dir, hospital_id="hospital_a")
            self.assertEqual(len(samples), 4)
            self.assertTrue(all("image" in s and "label" in s for s in samples))

    def test_validate_dataset_directory_missing_raises_error(self):
        """Verify FileNotFoundError when directory does not exist."""
        non_existent = Path("./non_existent_folder_xyz_123")
        with self.assertRaises(FileNotFoundError):
            validate_dataset_directory(non_existent, create_synthetic_if_empty=False)

    def test_validate_dataset_directory_bootstrap_synthetic(self):
        """Verify auto-generation of synthetic samples if folder is empty or missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = Path(tmp_dir) / "auto_bootstrap"
            samples = validate_dataset_directory(
                data_dir=target_dir,
                hospital_id="hospital_auto",
                create_synthetic_if_empty=True,
                num_synthetic_samples=3,
                spatial_size=(16, 16, 16),
            )
            self.assertEqual(len(samples), 3)
            self.assertTrue(target_dir.exists())

    def test_main_cli_dry_run(self):
        """Verify full CLI execution in dry-run mode returns exit code 0."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cli_args = [
                "--hospital-id", "hospital_test_node",
                "--data-dir", tmp_dir,
                "--create-synthetic",
                "--num-synthetic-samples", "4",
                "--roi-size", "16", "16", "16",
                "--epochs", "1",
                "--batch-size", "2",
                "--device", "cpu",
                "--dry-run",
            ]
            exit_code = main(cli_args)
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
