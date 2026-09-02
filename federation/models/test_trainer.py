"""
Unit tests for Loss Functions, Dice Metrics, and Local Client Trainer.
======================================================================
Tests Dice computation accuracy, loss decrease on training steps,
and multi-epoch local client execution.
"""

import sys
import unittest
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.models.metrics import (
    PyTorchDiceCELoss,
    PyTorchDiceLoss,
    compute_dice_score,
    get_loss_function,
)
from federation.models.trainer import (
    get_lr_scheduler,
    get_optimizer,
    train_epoch,
    train_local_client,
    validate,
)
from federation.models.unet3d import create_unet3d_model


class DictTensorDataset(torch.utils.data.Dataset):
    """Simple wrapper dataset yielding dictionary batches for trainer tests."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


class TestTrainerAndMetrics(unittest.TestCase):
    """Test suite for metrics calculation, losses, and local client trainer."""

    def setUp(self):
        self.in_channels = 4
        self.out_channels = 3
        self.spatial_dim = (16, 16, 16)
        self.batch_size = 2

        # Create synthetic tensors
        self.images = torch.randn(4, self.in_channels, *self.spatial_dim)
        # Binary multi-region targets
        self.labels = (torch.rand(4, self.out_channels, *self.spatial_dim) > 0.7).float()

        self.dataset = DictTensorDataset(self.images, self.labels)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dice_metric_perfect_and_zero_overlap(self):
        """Verify Dice score is 1.0 for perfect overlap and 0.0 for disjoint sets."""
        target = torch.zeros(1, 3, 8, 8, 8)
        target[:, :, 2:6, 2:6, 2:6] = 1.0

        # Perfect prediction (high logits corresponding to sigmoid > 0.5)
        perfect_pred = torch.full((1, 3, 8, 8, 8), -10.0)
        perfect_pred[:, :, 2:6, 2:6, 2:6] = 10.0

        perfect_metrics = compute_dice_score(perfect_pred, target, apply_sigmoid=True)
        self.assertAlmostEqual(perfect_metrics["dice_mean"], 1.0, places=2)
        self.assertAlmostEqual(perfect_metrics["dice_tc"], 1.0, places=2)

        # Disjoint prediction
        disjoint_pred = torch.full((1, 3, 8, 8, 8), -10.0)
        disjoint_pred[:, :, 0:2, 0:2, 0:2] = 10.0
        disjoint_metrics = compute_dice_score(disjoint_pred, target, apply_sigmoid=True)
        self.assertLess(disjoint_metrics["dice_mean"], 0.1)

    def test_loss_functions(self):
        """Verify DiceLoss and DiceCELoss compute positive scalar losses."""
        dice_loss = PyTorchDiceLoss()
        dice_ce_loss = PyTorchDiceCELoss()

        pred = torch.randn(2, 3, 8, 8, 8, requires_grad=True)
        target = (torch.rand(2, 3, 8, 8, 8) > 0.5).float()

        loss1 = dice_loss(pred, target)
        loss2 = dice_ce_loss(pred, target)

        self.assertTrue(torch.isfinite(loss1))
        self.assertTrue(torch.isfinite(loss2))
        self.assertGreater(loss1.item(), 0.0)
        self.assertGreater(loss2.item(), 0.0)

    def test_optimizer_and_scheduler(self):
        """Verify optimizer and scheduler construction."""
        model = create_unet3d_model(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=(4, 8, 16),
            strides=(2, 2),
        )
        opt = get_optimizer(model, optimizer_name="AdamW", lr=1e-3)
        self.assertIsInstance(opt, torch.optim.AdamW)

        scheduler = get_lr_scheduler(opt, scheduler_name="cosine", epochs=5)
        self.assertIsNotNone(scheduler)

    def test_single_epoch_training_reduces_loss(self):
        """Verify train_epoch executes and reduces loss on overfit batch."""
        model = create_unet3d_model(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=(4, 8, 16),
            strides=(2, 2),
            dropout=0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        loss_fn = PyTorchDiceCELoss()

        # Run 5 epochs on small loader to verify loss decreases
        first_loss = None
        last_loss = None
        for _ in range(5):
            metrics = train_epoch(model, self.loader, optimizer, loss_fn, gradient_clip_val=1.0)
            if first_loss is None:
                first_loss = metrics["train_loss"]
            last_loss = metrics["train_loss"]

        self.assertLess(last_loss, first_loss)
        self.assertEqual(metrics["num_samples"], 4)

    def test_validation_evaluation(self):
        """Verify validation function calculates val_loss and val_dice."""
        model = create_unet3d_model(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=(4, 8, 16),
            strides=(2, 2),
        )
        val_results = validate(model, self.loader)
        self.assertIn("val_loss", val_results)
        self.assertIn("val_dice_mean", val_results)
        self.assertTrue(0.0 <= val_results["val_dice_mean"] <= 1.0)

    def test_train_local_client_pipeline(self):
        """Verify complete multi-epoch local client training."""
        model = create_unet3d_model(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=(4, 8, 16),
            strides=(2, 2),
        )
        results = train_local_client(
            model=model,
            train_loader=self.loader,
            val_loader=self.loader,
            epochs=2,
            lr=1e-3,
        )

        self.assertIn("train_loss", results)
        self.assertIn("val_loss", results)
        self.assertIn("val_dice_mean", results)
        self.assertEqual(results["epochs_completed"], 2)
        self.assertEqual(results["num_samples"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
