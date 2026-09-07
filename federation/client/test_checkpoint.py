"""
Unit tests for FedMed Local Model Checkpointing and Parameter Diffing.
Tests parameter delta calculation, reconstruction, statistics, checkpoint saving/loading,
best model tracking, checkpoint rotation, and fallback recovery on failure.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.client.checkpoint import (
    ClientCheckpointManager,
    apply_parameter_delta,
    compute_delta_statistics,
    compute_parameter_delta,
)
from federation.client.fl_client import FedMedClient
from federation.models.unet3d import create_unet3d_model, get_model_parameters


class SimpleTensorDataset(Dataset):
    """Simple dictionary dataset for unit testing."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


class TestClientCheckpointAndDiffing(unittest.TestCase):
    """Test suite for checkpoint manager and parameter diffing utilities."""

    def setUp(self):
        self.in_channels = 4
        self.out_channels = 3
        self.spatial_dim = (16, 16, 16)

        self.model = create_unet3d_model(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=(4, 8, 16),
            strides=(2, 2),
            dropout=0.0,
        )

    def test_parameter_delta_computation_and_application(self):
        """Verify Delta W = W_local - W_global and W_new = W_base + Delta W."""
        global_params = get_model_parameters(self.model)

        # Create simulated local update by perturbing float weights
        local_params = [
            (p + 1) if np.issubdtype(p.dtype, np.integer) else (p + 0.25).astype(np.float32)
            for p in global_params
        ]

        # Compute delta
        deltas = compute_parameter_delta(local_params, global_params)
        self.assertEqual(len(deltas), len(global_params))

        # Reconstruct local params from baseline and deltas
        reconstructed = apply_parameter_delta(global_params, deltas)
        for loc, rec in zip(local_params, reconstructed):
            np.testing.assert_allclose(loc, rec, rtol=1e-5, atol=1e-5)

    def test_delta_statistics(self):
        """Verify delta summary statistics computation."""
        delta1 = np.zeros((4, 4), dtype=np.float32)
        delta1[0, 0] = 3.0
        delta1[0, 1] = 4.0

        stats = compute_delta_statistics([delta1])
        self.assertEqual(stats["total_elements"], 16)
        self.assertAlmostEqual(stats["l2_norm"], 5.0, places=4)  # sqrt(3^2 + 4^2) = 5
        self.assertAlmostEqual(stats["max_abs_update"], 4.0, places=4)
        self.assertAlmostEqual(stats["sparsity_ratio"], 14 / 16, places=4)

    def test_checkpoint_manager_save_and_load_latest(self):
        """Verify saving and loading latest checkpoint."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = ClientCheckpointManager(hospital_id="hosp_a", checkpoint_dir=tmp_dir)

            saved_path = mgr.save_latest(
                model=self.model,
                round_num=1,
                metrics={"train_loss": 0.45},
            )
            self.assertTrue(saved_path.exists())
            self.assertEqual(saved_path.name, "hosp_a_latest.pth")

            # Load into a fresh model
            fresh_model = create_unet3d_model(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                channels=(4, 8, 16),
                strides=(2, 2),
            )
            ckpt = mgr.load_latest(fresh_model)
            self.assertEqual(ckpt["round"], 1)
            self.assertEqual(ckpt["metrics"]["train_loss"], 0.45)

            # Verify parameters match
            for p1, p2 in zip(self.model.parameters(), fresh_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2, atol=1e-6))

    def test_checkpoint_manager_save_and_load_best(self):
        """Verify best model tracking across multiple rounds."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = ClientCheckpointManager(hospital_id="hosp_b", checkpoint_dir=tmp_dir)

            # Round 1: Dice 0.65 -> Saves best
            p1 = mgr.save_best(self.model, round_num=1, dice_score=0.65)
            self.assertIsNotNone(p1)
            self.assertEqual(mgr.best_dice, 0.65)

            # Round 2: Dice 0.55 -> Does not save (worse)
            p2 = mgr.save_best(self.model, round_num=2, dice_score=0.55)
            self.assertIsNone(p2)
            self.assertEqual(mgr.best_dice, 0.65)

            # Round 3: Dice 0.82 -> Saves new best
            p3 = mgr.save_best(self.model, round_num=3, dice_score=0.82)
            self.assertIsNotNone(p3)
            self.assertEqual(mgr.best_dice, 0.82)

            # Load best model
            fresh_model = create_unet3d_model(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                channels=(4, 8, 16),
                strides=(2, 2),
            )
            best_ckpt = mgr.load_best(fresh_model)
            self.assertEqual(best_ckpt["round"], 3)
            self.assertEqual(best_ckpt["metrics"]["best_dice"], 0.82)

    def test_checkpoint_rotation_keep_last_n(self):
        """Verify older round checkpoints are cleaned up when keep_last_n is exceeded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = ClientCheckpointManager(hospital_id="hosp_c", checkpoint_dir=tmp_dir, keep_last_n=2)

            for r in range(1, 6):
                mgr.save_latest(self.model, round_num=r)

            round_files = mgr.list_round_checkpoints()
            # Should only keep rounds 4 and 5
            self.assertEqual(len(round_files), 2)
            self.assertEqual(round_files[0].name, "hosp_c_round_004.pth")
            self.assertEqual(round_files[1].name, "hosp_c_round_005.pth")

    def test_client_fallback_recovery_on_training_failure(self):
        """Verify client gracefully recovers to global baseline on training failure without crashing."""
        images = torch.randn(4, self.in_channels, *self.spatial_dim)
        labels = (torch.rand(4, self.out_channels, *self.spatial_dim) > 0.7).float()
        loader = DataLoader(SimpleTensorDataset(images, labels), batch_size=2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            client = FedMedClient(
                hospital_id="hosp_recovery_test",
                model=self.model,
                train_loader=loader,
                config={"storage": {"checkpoint_dir": tmp_dir}},
                device="cpu",
            )

            global_params = client.get_parameters()

            # Mock train_local_client to simulate an unexpected CUDA Out-of-Memory or NaN error
            with patch("federation.client.fl_client.train_local_client", side_effect=RuntimeError("CUDA out of memory")):
                returned_params, num_samples, metrics = client.fit(
                    parameters=global_params,
                    config={"server_round": 1, "local_epochs": 1},
                )

                # Client should NOT crash and should return sample count 0 and failure status
                self.assertEqual(num_samples, 0)
                self.assertEqual(metrics["status"], "failed")
                self.assertIn("CUDA out of memory", metrics["error"])

                # Model weights should be restored to global parameter baseline
                restored_params = client.get_parameters()
                for p_glob, p_res in zip(global_params, restored_params):
                    np.testing.assert_allclose(p_glob, p_res, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
