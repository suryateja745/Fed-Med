"""
Unit tests for FedMed Flower Server and FedAvg Strategy.
Tests initial parameter generation, round configuration callbacks,
strategy instantiation, CLI argument parsing, and dry-run execution.
"""

import sys
import unittest
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.models.unet3d import create_unet3d_model
from federation.server.fl_server import (
    create_fedavg_strategy,
    get_evaluate_config_fn,
    get_fit_config_fn,
    get_initial_parameters,
)
from federation.server.run_server import main, parse_args


class TestFedMedServer(unittest.TestCase):
    """Test suite for Flower server coordinator and FedAvg strategy."""

    def setUp(self):
        self.model = create_unet3d_model(
            in_channels=4,
            out_channels=3,
            channels=(4, 8, 16),
            strides=(2, 2),
            dropout=0.0,
        )

    def test_get_initial_parameters(self):
        """Verify initial global parameters extraction from model."""
        params = get_initial_parameters(model=self.model)
        self.assertIsNotNone(params)
        self.assertTrue(hasattr(params, "tensors"))
        self.assertGreater(len(params.tensors), 0)

    def test_fit_and_evaluate_config_callbacks(self):
        """Verify round configuration callback generation."""
        custom_config = {
            "training": {
                "local_epochs": 4,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "optimizer": "AdamW",
                "loss_function": "DiceCELoss",
            }
        }

        fit_fn = get_fit_config_fn(custom_config)
        eval_fn = get_evaluate_config_fn(custom_config)

        # Test Round 1 fit config
        fit_cfg_r1 = fit_fn(1)
        self.assertEqual(fit_cfg_r1["server_round"], 1)
        self.assertEqual(fit_cfg_r1["local_epochs"], 4)
        self.assertEqual(fit_cfg_r1["learning_rate"], 1e-3)
        self.assertEqual(fit_cfg_r1["optimizer"], "AdamW")

        # Test Round 2 eval config
        eval_cfg_r2 = eval_fn(2)
        self.assertEqual(eval_cfg_r2["server_round"], 2)
        self.assertEqual(eval_cfg_r2["loss_function"], "DiceCELoss")

    def test_create_fedavg_strategy(self):
        """Verify FedAvg strategy construction with client thresholds."""
        strategy = create_fedavg_strategy(
            model=self.model,
            min_fit_clients=3,
            min_available_clients=3,
            min_evaluate_clients=2,
            fraction_fit=1.0,
        )

        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.min_fit_clients, 3)
        self.assertEqual(strategy.min_available_clients, 3)
        self.assertEqual(strategy.min_evaluate_clients, 2)
        self.assertEqual(strategy.fraction_fit, 1.0)
        self.assertIsNotNone(strategy.initial_parameters)

    def test_server_cli_argument_parsing(self):
        """Verify server CLI argument parsing."""
        args = parse_args([
            "--server-address", "127.0.0.1:9090",
            "--num-rounds", "15",
            "--min-fit-clients", "3",
            "--min-available-clients", "3",
            "--min-evaluate-clients", "2",
            "--dry-run",
        ])

        self.assertEqual(args.server_address, "127.0.0.1:9090")
        self.assertEqual(args.num_rounds, 15)
        self.assertEqual(args.min_fit_clients, 3)
        self.assertEqual(args.min_available_clients, 3)
        self.assertEqual(args.min_evaluate_clients, 2)
        self.assertTrue(args.dry_run)

    def test_run_server_main_dry_run(self):
        """Verify server CLI runner main execution in dry-run mode returns exit code 0."""
        exit_code = main([
            "--num-rounds", "5",
            "--min-fit-clients", "2",
            "--min-available-clients", "2",
            "--dry-run",
        ])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
