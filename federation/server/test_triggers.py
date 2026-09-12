"""
Unit test suite for AutoDispatchTrigger.
Tests automated model dispatch for offline admin mode, best/latest model resolution,
fallback mechanisms, and audit log generation.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.models.unet3d import (
    create_unet3d_model,
    get_model_parameters,
    save_model_checkpoint,
)
from federation.server.model_manager import GlobalModelManager
from federation.server.triggers import AutoDispatchTrigger, DispatchAuditEvent


class TestAutoDispatchTrigger(unittest.TestCase):
    """Test suite for AutoDispatchTrigger in unattended/offline admin coordinator mode."""

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
        self.audit_log = Path(self.temp_dir) / "logs" / "dispatch_audit.jsonl"
        self.model = self._create_model()

        self.model_manager = GlobalModelManager(
            checkpoint_dir=self.checkpoint_dir,
            model=self.model,
            export_best=True,
        )
        self.trigger = AutoDispatchTrigger(
            model_manager=self.model_manager,
            checkpoint_dir=self.checkpoint_dir,
            audit_log_file=self.audit_log,
            prefer_best=True,
        )

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dispatch_preset_baseline_when_no_checkpoints(self):
        """Test that baseline weights are automatically served if no training rounds occurred."""
        params, meta = self.trigger.handle_client_connect(
            hospital_id="hospital_a",
            client_info={"os": "Windows", "gpu": "RTX 2050"},
        )

        self.assertIsNotNone(params)
        self.assertGreater(len(params), 0)
        self.assertEqual(meta["hospital_id"], "hospital_a")
        self.assertEqual(meta["model_version"], "preset_baseline_round_0")
        self.assertEqual(meta["round_num"], 0)

        # Verify audit log
        self.assertTrue(self.audit_log.exists())
        history = self.trigger.get_dispatch_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["hospital_id"], "hospital_a")
        self.assertEqual(history[0]["status"], "SUCCESS")

    def test_dispatch_best_global_model(self):
        """Test that trigger automatically serves best_global_model.pth when available."""
        # Save Round 1: Dice = 0.60
        self.model_manager.save_round_checkpoint(
            round_num=1,
            model=self.model,
            metrics={"val_dice_mean": 0.60},
            participating_clients=["hospital_a"],
        )
        # Save Round 2: Dice = 0.85 (New Best)
        self.model_manager.save_round_checkpoint(
            round_num=2,
            model=self.model,
            metrics={"val_dice_mean": 0.85},
            participating_clients=["hospital_a", "hospital_b"],
        )

        params, meta = self.trigger.handle_client_connect(
            hospital_id="hospital_c",
            client_info={"dataset_size": 25},
        )

        self.assertEqual(meta["model_version"], "best_global_model")
        self.assertEqual(meta["round_num"], 2)
        self.assertAlmostEqual(meta["dice_score"], 0.85, places=4)
        self.assertIn("best_global_model.pth", meta["checkpoint_path"])

    def test_multiple_hospital_dispatches_and_stats(self):
        """Test multiple client requests and aggregate statistics tracking."""
        self.trigger.handle_client_connect("hospital_a")
        self.trigger.handle_client_connect("hospital_b")
        self.trigger.handle_client_connect("hospital_a")

        stats = self.trigger.get_dispatch_stats()
        self.assertEqual(stats["total_dispatches"], 3)
        self.assertEqual(stats["unique_hospitals_served"], 2)
        self.assertEqual(stats["hospitals_served_breakdown"]["hospital_a"], 2)
        self.assertEqual(stats["hospitals_served_breakdown"]["hospital_b"], 1)
        self.assertIsNotNone(stats["last_dispatch"])

        history = self.trigger.get_dispatch_history(limit=2)
        self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
