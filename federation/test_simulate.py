"""
Unit test suite for FedMed Multi-Hospital Simulation Testbed.
Tests simulation dataset generation, client factory, convergence plot generation,
and end-to-end multi-round execution.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.simulate import (
    build_client_factory,
    generate_simulation_plots,
    run_simulation,
    setup_simulation_environment,
)


class TestSimulationTestbed(unittest.TestCase):
    """Test suite for simulation engine, client factory, and plotting."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "sim_data"
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.plot_dir = Path(self.temp_dir) / "reports"

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_setup_simulation_environment(self):
        """Test partitioning and generation of simulated hospital datasets."""
        partitions, hospital_map = setup_simulation_environment(
            num_clients=3,
            data_dir=self.data_dir,
            partition_type="quantity_skew",
            num_samples_per_client=2,
            spatial_size=(16, 16, 16),
        )

        self.assertEqual(len(partitions), 3)
        self.assertIn("hospital_a", partitions)
        self.assertIn("hospital_b", partitions)
        self.assertIn("hospital_c", partitions)
        self.assertEqual(hospital_map.get("0"), "hospital_a")
        self.assertEqual(hospital_map.get("1"), "hospital_b")
        self.assertEqual(hospital_map.get("2"), "hospital_c")

    def test_generate_simulation_plots(self):
        """Test generation of convergence PNG images."""
        mock_history = {
            "rounds": [1, 2, 3],
            "train_loss": [1.5, 1.2, 0.9],
            "val_loss": [1.6, 1.3, 1.0],
            "val_dice_mean": [0.35, 0.55, 0.72],
            "val_dice_tc": [0.30, 0.50, 0.68],
            "val_dice_wt": [0.40, 0.60, 0.78],
            "val_dice_et": [0.25, 0.45, 0.62],
        }

        plot_paths = generate_simulation_plots(
            history=mock_history,
            output_dir=self.plot_dir,
        )

        self.assertIn("round_vs_dice", plot_paths)
        self.assertIn("round_vs_loss", plot_paths)
        self.assertTrue(plot_paths["round_vs_dice"].exists())
        self.assertTrue(plot_paths["round_vs_loss"].exists())
        self.assertGreater(plot_paths["round_vs_dice"].stat().st_size, 1000)
        self.assertGreater(plot_paths["round_vs_loss"].stat().st_size, 1000)

    def test_run_simulation_dry_run(self):
        """Test end-to-end dry run simulation execution."""
        summary = run_simulation(
            num_clients=2,
            num_rounds=1,
            epochs=1,
            batch_size=1,
            data_dir=self.data_dir,
            checkpoint_dir=self.checkpoint_dir,
            plot_dir=self.plot_dir,
            device="cpu",
            dry_run=True,
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary["num_clients"], 2)
        self.assertEqual(summary["num_rounds"], 1)
        self.assertTrue((self.plot_dir / "simulation_summary.json").exists())
        self.assertTrue((self.plot_dir / "round_vs_dice.png").exists())
        self.assertTrue((self.plot_dir / "round_vs_loss.png").exists())
        self.assertTrue((self.checkpoint_dir / "global_model_latest.pth").exists())


if __name__ == "__main__":
    unittest.main()
