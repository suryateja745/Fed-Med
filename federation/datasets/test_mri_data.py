"""
Unit tests for MRI Transforms, Hospital Partitioners, and DataLoaders.
Tests local directory ingestion, IID/Non-IID partitioning, MONAI transforms,
and PyTorch DataLoader batch generation.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.datasets.mri_transforms import (
    ConvertToMultiChannelBasedOnBratsClassesd,
    SimpleDictTransform,
    get_train_transforms,
    get_val_transforms,
)
from federation.datasets.partitioner import (
    FedMedMRIDataset,
    create_hospital_dataloaders,
    generate_synthetic_mri_dataset,
    partition_dataset,
    scan_local_dataset,
)


class TestMRIDatasetPipeline(unittest.TestCase):
    """Test suite for data loading, partitioning, and transforms."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        # Create 6 synthetic samples: [4, 32, 32, 32] images, [3, 32, 32, 32] labels
        self.samples = generate_synthetic_mri_dataset(
            output_dir=self.data_dir,
            num_samples=6,
            spatial_size=(32, 32, 32),
            in_channels=4,
            out_channels=3,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_synthetic_data_generation(self):
        """Verify synthetic generator creates expected files."""
        self.assertEqual(len(self.samples), 6)
        for s in self.samples:
            self.assertTrue(Path(s["image"]).exists())
            self.assertTrue(Path(s["label"]).exists())

    def test_scan_local_dataset(self):
        """Verify directory scanner finds all paired samples."""
        scanned = scan_local_dataset(self.data_dir)
        self.assertEqual(len(scanned), 6)
        for item in scanned:
            self.assertIn("image", item)
            self.assertIn("label", item)

    def test_partition_dataset_iid(self):
        """Verify IID partitioning across 3 hospitals."""
        partitions = partition_dataset(self.samples, num_clients=3, partition_type="iid")
        self.assertIn("hospital_a", partitions)
        self.assertIn("hospital_b", partitions)
        self.assertIn("hospital_c", partitions)
        total_partitioned = sum(len(p) for p in partitions.values())
        self.assertEqual(total_partitioned, 6)

    def test_partition_dataset_non_iid(self):
        """Verify quantity-skew and dirichlet partitioning."""
        partitions_skew = partition_dataset(self.samples, num_clients=3, partition_type="quantity_skew")
        self.assertEqual(sum(len(p) for p in partitions_skew.values()), 6)

        partitions_dirichlet = partition_dataset(self.samples, num_clients=3, partition_type="dirichlet", alpha=0.3)
        self.assertEqual(sum(len(p) for p in partitions_dirichlet.values()), 6)

    def test_brats_label_conversion(self):
        """Verify conversion from BraTS classes (1, 2, 4) to 3 binary channels."""
        converter = ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"])
        # Mock label with BraTS integer annotations
        raw_seg = np.zeros((1, 16, 16, 16), dtype=np.int32)
        raw_seg[0, 2:5, 2:5, 2:5] = 1  # Necrotic/Non-enhancing
        raw_seg[0, 5:8, 5:8, 5:8] = 2  # Edema
        raw_seg[0, 8:12, 8:12, 8:12] = 4  # Enhancing tumor

        out = converter({"label": raw_seg})
        converted = out["label"]
        self.assertEqual(converted.shape[0], 3)  # 3 channels: TC, WT, ET
        # WT (channel 1) should have elements from 1, 2, and 4
        self.assertGreater(converted[1].sum(), converted[0].sum())
        # ET (channel 2) should only have elements from 4
        self.assertEqual(converted[2].sum(), (raw_seg == 4).sum())

    def test_transforms_and_dataloader(self):
        """Verify end-to-end hospital DataLoader batch output shapes."""
        train_loader, val_loader, n_train, n_val = create_hospital_dataloaders(
            hospital_id="hospital_a",
            data_dir=self.data_dir,
            batch_size=2,
            train_val_split=0.66,
            roi_size=(32, 32, 32),
        )

        self.assertGreater(n_train, 0)
        self.assertGreater(n_val, 0)

        # Fetch one batch from train_loader
        batch = next(iter(train_loader))
        images = batch["image"]
        labels = batch["label"]

        self.assertEqual(images.shape[0], 2)  # Batch size
        self.assertEqual(images.shape[1], 4)  # 4 MRI modalities
        self.assertEqual(images.shape[2:], (32, 32, 32))  # Spatial dimensions

        self.assertEqual(labels.shape[0], 2)  # Batch size
        self.assertEqual(labels.shape[1], 3)  # 3 tumor classes
        self.assertEqual(labels.shape[2:], (32, 32, 32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
