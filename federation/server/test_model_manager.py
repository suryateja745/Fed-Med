"""
Unit test suite for GlobalModelManager.
Tests global model checkpointing, versioning, metadata logging, best model tracking,
and parameter loading into fresh MONAI 3D U-Net instances.
"""

import json
import shutil
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

from federation.models.unet3d import (
    create_unet3d_model,
    get_model_parameters,
    set_model_parameters,
)
from federation.server.model_manager import GlobalModelManager


class TestGlobalModelManager(unittest.TestCase):
    """Test suite for GlobalModelManager checkpointing and state tracking."""

    def _create_model(self):
        return create_unet3d_model(
            in_channels=4,
            out_channels=3,
            channels=(8, 16),
            strides=(2,),
            num_res_units=1,
            norm="instance",
            dropout=0.0,
        )

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.model = self._create_model()

        self.manager = GlobalModelManager(
            checkpoint_dir=self.checkpoint_dir,
            model=self.model,
            export_best=True,
            keep_last_n=3,
        )

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initial_metadata_registry_creation(self):
        """Test that global_model_metadata.json is properly initialized."""
        metadata_path = self.checkpoint_dir / "global_model_metadata.json"
        self.assertTrue(metadata_path.exists())

        metadata = self.manager.get_metadata()
        self.assertEqual(metadata.get("project"), "FedMed")
        self.assertEqual(metadata.get("latest_round"), 0)
        self.assertEqual(metadata.get("best_dice"), -1.0)
        self.assertEqual(len(metadata.get("rounds_history", [])), 0)

    def test_save_round_checkpoint_and_load_latest(self):
        """Test saving a round checkpoint and loading it into a fresh model instance."""
        params = get_model_parameters(self.model)
        # Modify first parameter array slightly to create a signature
        params[0] = params[0] + 0.5
        set_model_parameters(self.model, params)

        metrics = {"val_loss": 0.42, "val_dice_mean": 0.75, "train_loss": 0.50}
        clients = ["hospital_a", "hospital_b"]

        latest_path = self.manager.save_round_checkpoint(
            round_num=1,
            model=self.model,
            metrics=metrics,
            participating_clients=clients,
        )

        self.assertTrue(latest_path.exists())
        self.assertTrue((self.checkpoint_dir / "global_model_round_001.pth").exists())
        self.assertTrue((self.checkpoint_dir / "best_global_model.pth").exists())

        # Verify metadata
        metadata = self.manager.get_metadata()
        self.assertEqual(metadata["latest_round"], 1)
        self.assertEqual(metadata["best_round"], 1)
        self.assertAlmostEqual(metadata["best_dice"], 0.75, places=4)
        self.assertEqual(len(metadata["rounds_history"]), 1)

        history_entry = metadata["rounds_history"][0]
        self.assertEqual(history_entry["round"], 1)
        self.assertEqual(history_entry["participating_clients"], clients)
        self.assertAlmostEqual(history_entry["global_dice_score"], 0.75, places=4)
        self.assertTrue(history_entry["is_best"])

        # Fresh model instance
        fresh_model = self._create_model()
        loaded_model, ckpt_dict = self.manager.load_latest(fresh_model)
        loaded_params = get_model_parameters(loaded_model)

        np.testing.assert_allclose(params[0], loaded_params[0], rtol=1e-5, atol=1e-6)
        self.assertEqual(ckpt_dict["round"], 1)

    def test_best_model_tracking(self):
        """Test that best_global_model.pth is only overwritten when Dice improves."""
        # Round 1: Dice = 0.65
        self.manager.save_round_checkpoint(
            round_num=1,
            model=self.model,
            metrics={"val_dice_mean": 0.65},
            participating_clients=["hospital_a"],
        )
        self.assertEqual(self.manager.best_round, 1)
        self.assertAlmostEqual(self.manager.best_dice, 0.65, places=4)

        # Round 2: Dice = 0.60 (Worse -> Should NOT update best)
        self.manager.save_round_checkpoint(
            round_num=2,
            model=self.model,
            metrics={"val_dice_mean": 0.60},
            participating_clients=["hospital_a", "hospital_b"],
        )
        self.assertEqual(self.manager.best_round, 1)
        self.assertAlmostEqual(self.manager.best_dice, 0.65, places=4)

        # Round 3: Dice = 0.82 (Better -> Should update best)
        self.manager.save_round_checkpoint(
            round_num=3,
            model=self.model,
            metrics={"val_dice_mean": 0.82},
            participating_clients=["hospital_b", "hospital_c"],
        )
        self.assertEqual(self.manager.best_round, 3)
        self.assertAlmostEqual(self.manager.best_dice, 0.82, places=4)

        # Load best model and verify metadata
        fresh_model = self._create_model()
        _, best_ckpt = self.manager.load_best(fresh_model)
        self.assertEqual(best_ckpt["round"], 3)
        self.assertAlmostEqual(best_ckpt["metrics"]["best_dice"], 0.82, places=4)

    def test_checkpoint_rotation_keep_last_n(self):
        """Test that intermediate round checkpoints are rotated according to keep_last_n."""
        # Manager is configured with keep_last_n=3
        for r in range(1, 6):
            self.manager.save_round_checkpoint(
                round_num=r,
                model=self.model,
                metrics={"val_dice_mean": 0.5 + r * 0.05},
            )

        round_files = self.manager.list_round_checkpoints()
        # Should retain only the last 3 rounds: round 3, 4, 5
        self.assertEqual(len(round_files), 3)
        round_names = [f.name for f in round_files]
        self.assertIn("global_model_round_003.pth", round_names)
        self.assertIn("global_model_round_004.pth", round_names)
        self.assertIn("global_model_round_005.pth", round_names)
        self.assertNotIn("global_model_round_001.pth", round_names)
        self.assertNotIn("global_model_round_002.pth", round_names)

        # Latest and best must still exist
        self.assertTrue((self.checkpoint_dir / "global_model_latest.pth").exists())
        self.assertTrue((self.checkpoint_dir / "best_global_model.pth").exists())

    def test_load_specific_round(self):
        """Test loading a specific round checkpoint."""
        self.manager.save_round_checkpoint(
            round_num=1,
            model=self.model,
            metrics={"val_dice_mean": 0.70},
        )
        fresh_model = self._create_model()
        loaded_model, ckpt = self.manager.load_round(round_num=1, model=fresh_model)
        self.assertEqual(ckpt["round"], 1)

        # Testing non-existent round raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.manager.load_round(round_num=99, model=fresh_model)

    def test_update_evaluation_metrics(self):
        """Test updating evaluation metrics post-round."""
        self.manager.save_round_checkpoint(
            round_num=1,
            model=self.model,
            metrics={"train_loss": 0.5},
            participating_clients=["hosp_1"],
        )

        eval_metrics = {"val_dice_mean": 0.88, "val_dice_tc": 0.85, "val_dice_wt": 0.90}
        self.manager.update_evaluation_metrics(round_num=1, metrics=eval_metrics)

        metadata = self.manager.get_metadata()
        self.assertEqual(metadata["best_round"], 1)
        self.assertAlmostEqual(metadata["best_dice"], 0.88, places=4)

        entry = metadata["rounds_history"][0]
        self.assertAlmostEqual(entry["global_dice_score"], 0.88, places=4)
        self.assertEqual(entry["metrics"]["val_dice_tc"], 0.85)


if __name__ == "__main__":
    unittest.main()
