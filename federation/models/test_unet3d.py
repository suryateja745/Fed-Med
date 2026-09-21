"""
Unit tests for 3D U-Net Model Architecture and Utilities.
Tests forward pass on synthetic 3D MRI tensors, parameter extraction for Flower,
and checkpoint serialization.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.models.unet3d import (
    FedMedUNet3D,
    PyTorchUNet3D,
    build_unet3d_from_config,
    count_parameters,
    create_unet3d_model,
    get_model_parameters,
    load_model_checkpoint,
    save_model_checkpoint,
    set_model_parameters,
)


class TestUNet3DArchitecture(unittest.TestCase):
    """Test suite for 3D U-Net model and Flower utility methods."""

    def setUp(self):
        """Setup standard test configurations."""
        self.in_channels = 4
        self.out_channels = 3
        self.batch_size = 1
        self.spatial_dim = (32, 32, 32)  # [D, H, W]

    def test_forward_pass_multi_modal(self):
        """Verify forward pass on synthetic 4-channel multi-modal 3D MRI tensor."""
        model = create_unet3d_model(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=(8, 16, 32, 64),
            strides=(2, 2, 2),
            num_res_units=1,
            dropout=0.0,
        )
        model.eval()

        # Synthetic 3D tensor: [B, C, D, H, W]
        x = torch.randn(self.batch_size, self.in_channels, *self.spatial_dim)
        with torch.no_grad():
            output = model(x)

        expected_shape = (self.batch_size, self.out_channels, *self.spatial_dim)
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Expected output shape {expected_shape}, got {output.shape}",
        )
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")

    def test_forward_pass_single_channel(self):
        """Verify forward pass on single-channel 3D MRI input."""
        model = create_unet3d_model(
            in_channels=1,
            out_channels=1,
            channels=(8, 16, 32),
            strides=(2, 2),
            num_res_units=1,
            dropout=0.0,
        )
        model.eval()

        x = torch.randn(2, 1, 16, 16, 16)
        with torch.no_grad():
            output = model(x)

        self.assertEqual(output.shape, (2, 1, 16, 16, 16))

    def test_pytorch_native_fallback(self):
        """Verify native PyTorch UNet3D forward pass."""
        model = PyTorchUNet3D(
            in_channels=4,
            out_channels=3,
            channels=(8, 16, 32),
            dropout=0.0,
        )
        model.eval()
        x = torch.randn(1, 4, 16, 16, 16)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape, (1, 3, 16, 16, 16))

    def test_parameter_count(self):
        """Verify parameter counting utility."""
        model = create_unet3d_model(
            in_channels=4,
            out_channels=3,
            channels=(8, 16, 32),
            strides=(2, 2),
        )
        counts = count_parameters(model)
        self.assertIn("trainable_parameters", counts)
        self.assertIn("total_parameters", counts)
        self.assertGreater(counts["trainable_parameters"], 0)
        self.assertEqual(counts["trainable_parameters"], counts["total_parameters"])

    def test_flower_parameter_exchange(self):
        """Verify extraction and injection of NumPy model parameters for Flower."""
        model1 = create_unet3d_model(
            in_channels=4,
            out_channels=3,
            channels=(8, 16, 32),
            strides=(2, 2),
        )
        model2 = create_unet3d_model(
            in_channels=4,
            out_channels=3,
            channels=(8, 16, 32),
            strides=(2, 2),
        )

        # Extract weights from model1
        params = get_model_parameters(model1)
        self.assertIsInstance(params, list)
        self.assertIsInstance(params[0], np.ndarray)

        # Load weights into model2
        set_model_parameters(model2, params)

        # Compare state_dicts to ensure exact match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2, atol=1e-6))

    def test_checkpoint_save_and_load(self):
        """Verify checkpoint saving and loading roundtrip."""
        import tempfile
        model = create_unet3d_model(
            in_channels=4,
            out_channels=3,
            channels=(8, 16, 32),
            strides=(2, 2),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_checkpoint.pth"
            metrics = {"dice_score": 0.854, "loss": 0.12}
            save_model_checkpoint(model, save_path, round_num=3, metrics=metrics)
            self.assertTrue(save_path.exists())

            loaded_model = create_unet3d_model(
                in_channels=4,
                out_channels=3,
                channels=(8, 16, 32),
                strides=(2, 2),
            )
            ckpt = load_model_checkpoint(loaded_model, save_path)
            self.assertEqual(ckpt["round"], 3)
            self.assertEqual(ckpt["metrics"]["dice_score"], 0.854)

            for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2, atol=1e-6))

    def test_build_from_config(self):
        """Verify model instantiation from fl_config.yaml."""
        model = build_unet3d_from_config()
        self.assertIsInstance(model, FedMedUNet3D)
        self.assertEqual(model.in_channels, 4)
        self.assertEqual(model.out_channels, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
