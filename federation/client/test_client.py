"""
Unit tests for FedMed Flower NumPyClient.
Tests get_parameters, set_parameters, local fit round execution,
validation evaluate round, and factory client creation.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.client.fl_client import FedMedClient, create_client
from federation.datasets.partitioner import generate_synthetic_mri_dataset
from federation.models.unet3d import create_unet3d_model, get_model_parameters


class SimpleTensorDataset(Dataset):
    """Simple dictionary dataset for client testing."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


class TestFedMedClient(unittest.TestCase):
    """Unit test suite for FedMed Flower NumPyClient."""

    def setUp(self):
        self.in_channels = 4
        self.out_channels = 3
        self.spatial_dim = (16, 16, 16)
        self.batch_size = 2

        # Create lightweight 3D U-Net for quick testing
        self.model = create_unet3d_model(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=(4, 8, 16),
            strides=(2, 2),
            dropout=0.0,
        )

        # Create synthetic batches
        images = torch.randn(4, self.in_channels, *self.spatial_dim)
        labels = (torch.rand(4, self.out_channels, *self.spatial_dim) > 0.7).float()

        self.train_dataset = SimpleTensorDataset(images, labels)
        self.val_dataset = SimpleTensorDataset(images[:2], labels[:2])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        self.client = FedMedClient(
            hospital_id="hospital_test",
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device="cpu",
        )

    def test_get_and_set_parameters(self):
        """Verify parameter extraction and injection without shape mismatches."""
        initial_params = self.client.get_parameters()
        self.assertIsInstance(initial_params, list)
        self.assertGreater(len(initial_params), 0)
        self.assertTrue(all(isinstance(p, np.ndarray) for p in initial_params))

        # Perturb parameters
        perturbed_params = [p + 0.5 for p in initial_params]
        self.client.set_parameters(perturbed_params)

        # Re-extract and verify values match perturbed parameters
        new_params = self.client.get_parameters()
        for p_orig, p_new in zip(perturbed_params, new_params):
            np.testing.assert_allclose(p_orig, p_new, rtol=1e-5, atol=1e-5)

    def test_get_properties(self):
        """Verify client properties reporting."""
        props = self.client.get_properties()
        self.assertEqual(props["hospital_id"], "hospital_test")
        self.assertEqual(props["device"], "cpu")
        self.assertEqual(props["num_train_samples"], 4)
        self.assertEqual(props["num_val_samples"], 2)

    def test_fit_round_execution(self):
        """Verify local fit round executes and returns updated weights and metrics."""
        params = self.client.get_parameters()
        config = {
            "server_round": 1,
            "local_epochs": 2,
            "learning_rate": 1e-3,
            "optimizer": "AdamW",
            "loss_function": "DiceCELoss",
        }

        updated_params, num_samples, metrics = self.client.fit(params, config)

        self.assertEqual(len(updated_params), len(params))
        self.assertEqual(num_samples, 4)
        self.assertEqual(metrics["hospital_id"], "hospital_test")
        self.assertIn("train_loss", metrics)
        self.assertEqual(metrics["epochs_completed"], 2)
        self.assertIn("val_loss", metrics)
        self.assertIn("val_dice_mean", metrics)

    def test_evaluate_round_execution(self):
        """Verify evaluate round returns validation loss and Dice score."""
        params = self.client.get_parameters()
        config = {"loss_function": "DiceCELoss"}

        val_loss, num_samples, metrics = self.client.evaluate(params, config)

        self.assertIsInstance(val_loss, float)
        self.assertEqual(num_samples, 2)
        self.assertEqual(metrics["hospital_id"], "hospital_test")
        self.assertIn("val_dice_mean", metrics)
        self.assertTrue(0.0 <= metrics["val_dice_mean"] <= 1.0)
        self.assertIn("val_dice_tc", metrics)
        self.assertIn("val_dice_wt", metrics)
        self.assertIn("val_dice_et", metrics)

    def test_create_client_factory_on_synthetic_data(self):
        """Verify create_client factory with on-disk synthetic data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            generate_synthetic_mri_dataset(
                output_dir=tmp_dir,
                num_samples=4,
                spatial_size=(16, 16, 16),
                in_channels=4,
                out_channels=3,
            )

            client = create_client(
                hospital_id="hospital_factory",
                data_dir=tmp_dir,
                model=self.model,
                batch_size=2,
                roi_size=(16, 16, 16),
                device="cpu",
            )

            self.assertEqual(client.hospital_id, "hospital_factory")
            self.assertGreater(len(client.train_loader), 0)

            # Test a mini fit round on the factory-created client
            params = client.get_parameters()
            updated_params, num_samples, metrics = client.fit(
                params,
                config={"server_round": 1, "local_epochs": 1, "learning_rate": 1e-3},
            )
            self.assertEqual(len(updated_params), len(params))
            self.assertGreater(num_samples, 0)
            self.assertIn("train_loss", metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
