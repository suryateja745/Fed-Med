"""
Unit tests for Custom FedMedStrategy and Aggregation Functions.
Tests weighted parameter averaging, fit metrics aggregation, evaluate metrics aggregation,
Dice-weighted client fit aggregation, global evaluate aggregation, and centralized server evaluation.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.models.unet3d import create_unet3d_model, get_model_parameters
from federation.server.strategy import (
    ClientProxy,
    EvaluateRes,
    FedMedStrategy,
    FitRes,
    aggregate_evaluate_metrics,
    aggregate_fit_metrics,
    aggregate_weighted_parameters,
    create_fedmed_strategy,
    get_server_eval_fn,
    ndarrays_to_parameters,
)


class MockDataset(Dataset):
    """Simple dictionary dataset for centralized evaluation testing."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3, num_samples: int = 4):
        self.images = torch.randn(num_samples, in_channels, 16, 16, 16)
        self.labels = (torch.rand(num_samples, out_channels, 16, 16, 16) > 0.7).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


class TestFedMedStrategy(unittest.TestCase):
    """Test suite for custom FedMedStrategy and aggregation math."""

    def setUp(self):
        self.model = create_unet3d_model(
            in_channels=4,
            out_channels=3,
            channels=(4, 8, 16),
            strides=(2, 2),
            dropout=0.0,
        )

    def test_aggregate_weighted_parameters(self):
        """Verify weighted average computation across client parameter arrays."""
        p1 = [np.array([1.0, 2.0], dtype=np.float32), np.array([10.0], dtype=np.float32)]
        p2 = [np.array([3.0, 6.0], dtype=np.float32), np.array([30.0], dtype=np.float32)]

        # Client 1 weight: 1.0, Client 2 weight: 3.0 (Total: 4.0)
        # Expected: 0.25 * [1.0, 2.0] + 0.75 * [3.0, 6.0] = [2.5, 5.0]
        # Expected: 0.25 * [10.0] + 0.75 * [30.0] = [25.0]
        results = [(p1, 1.0), (p2, 3.0)]
        agg = aggregate_weighted_parameters(results)

        np.testing.assert_allclose(agg[0], np.array([2.5, 5.0], dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(agg[1], np.array([25.0], dtype=np.float32), atol=1e-5)

    def test_aggregate_fit_metrics(self):
        """Verify weighted average of training loss and metrics."""
        fit_metrics = [
            (10, {"train_loss": 0.5, "epoch_duration": 1.2, "round_duration": 3.6}),
            (30, {"train_loss": 0.1, "epoch_duration": 1.4, "round_duration": 4.2}),
        ]
        # Weighted loss: (10*0.5 + 30*0.1) / 40 = 8.0 / 40 = 0.2
        agg = aggregate_fit_metrics(fit_metrics)

        self.assertEqual(agg["total_fit_samples"], 40)
        self.assertEqual(agg["num_clients_fit"], 2)
        self.assertAlmostEqual(agg["train_loss"], 0.2, places=4)
        self.assertAlmostEqual(agg["avg_epoch_duration"], 1.3, places=4)

    def test_aggregate_evaluate_metrics(self):
        """Verify sample-weighted aggregation of multi-region Dice scores."""
        eval_metrics = [
            (20, {"dice_score": 0.80, "val_dice_tc": 0.75, "val_dice_wt": 0.85, "val_dice_et": 0.70}),
            (80, {"dice_score": 0.90, "val_dice_tc": 0.85, "val_dice_wt": 0.95, "val_dice_et": 0.80}),
        ]
        # Weighted dice: (20*0.80 + 80*0.90) / 100 = (16 + 72) / 100 = 0.88
        agg = aggregate_evaluate_metrics(eval_metrics)

        self.assertEqual(agg["total_eval_samples"], 100)
        self.assertAlmostEqual(agg["dice_score"], 0.88, places=4)
        self.assertAlmostEqual(agg["val_dice_tc"], 0.83, places=4)
        self.assertAlmostEqual(agg["val_dice_wt"], 0.93, places=4)
        self.assertAlmostEqual(agg["val_dice_et"], 0.78, places=4)

    def test_fedmed_strategy_aggregate_fit(self):
        """Verify FedMedStrategy aggregate_fit with Dice-weighted scoring."""
        strategy = create_fedmed_strategy(
            model=self.model,
            weighted_by_dice=True,
            min_fit_clients=2,
            min_available_clients=2,
        )

        params1 = get_model_parameters(self.model)
        params2 = [(p + 0.1) if np.issubdtype(p.dtype, np.floating) else p for p in params1]

        fit_res_1 = FitRes(
            parameters=ndarrays_to_parameters(params1),
            num_examples=10,
            metrics={"train_loss": 0.4, "dice_score": 0.8, "epoch_duration": 1.0, "round_duration": 3.0},
        )
        fit_res_2 = FitRes(
            parameters=ndarrays_to_parameters(params2),
            num_examples=10,
            metrics={"train_loss": 0.2, "dice_score": 0.9, "epoch_duration": 1.0, "round_duration": 3.0},
        )

        results = [(ClientProxy("c1"), fit_res_1), (ClientProxy("c2"), fit_res_2)]
        agg_params, agg_metrics = strategy.aggregate_fit(server_round=1, results=results, failures=[])

        self.assertIsNotNone(agg_params)
        self.assertEqual(agg_metrics["server_round"], 1)
        self.assertEqual(agg_metrics["total_fit_samples"], 20)
        self.assertIn("train_loss", agg_metrics)

    def test_fedmed_strategy_aggregate_evaluate(self):
        """Verify FedMedStrategy aggregate_evaluate tracks global best Dice."""
        strategy = create_fedmed_strategy(
            model=self.model,
            min_evaluate_clients=2,
            min_available_clients=2,
        )

        eval_res_1 = EvaluateRes(
            loss=0.35,
            num_examples=10,
            metrics={"val_dice_mean": 0.75, "val_dice_tc": 0.70, "val_dice_wt": 0.80, "val_dice_et": 0.65},
        )
        eval_res_2 = EvaluateRes(
            loss=0.25,
            num_examples=10,
            metrics={"val_dice_mean": 0.85, "val_dice_tc": 0.80, "val_dice_wt": 0.90, "val_dice_et": 0.75},
        )

        results = [(ClientProxy("c1"), eval_res_1), (ClientProxy("c2"), eval_res_2)]
        loss, metrics = strategy.aggregate_evaluate(server_round=1, results=results, failures=[])

        self.assertAlmostEqual(loss, 0.30, places=4)
        self.assertAlmostEqual(metrics["val_dice_mean"], 0.80, places=4)
        self.assertTrue(metrics.get("is_best", False))
        self.assertEqual(strategy.best_global_dice, 0.80)

    def test_centralized_server_evaluation_hook(self):
        """Verify centralized server benchmark evaluation against holdout dataset."""
        dataset = MockDataset(num_samples=4)
        val_loader = DataLoader(dataset, batch_size=2)

        eval_fn = get_server_eval_fn(
            model=self.model,
            val_loader=val_loader,
            device="cpu",
        )

        params = ndarrays_to_parameters(get_model_parameters(self.model))
        loss, metrics = eval_fn(server_round=1, parameters=params, config={})

        self.assertIsInstance(loss, float)
        self.assertIn("dice_mean", metrics)
        self.assertIn("dice_tc", metrics)
        self.assertIn("dice_wt", metrics)
        self.assertIn("dice_et", metrics)
        self.assertTrue(0.0 <= metrics["dice_mean"] <= 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
