"""
Unit test suite for FedMed Automated Triggers.
Tests AutoDispatchTrigger (offline model serving) and AutoAggregateTrigger (threshold-based
automated aggregation, checkpoint generation, round advancement, and event callbacks).
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
from federation.server.strategy import FedMedStrategy
from federation.server.sync_manager import RoundSyncManager
from federation.server.triggers import (
    AutoAggregateTrigger,
    AutoDispatchTrigger,
    DispatchAuditEvent,
    RoundCompleteEvent,
)


class TestAutoTriggers(unittest.TestCase):
    """Test suite for AutoDispatchTrigger and AutoAggregateTrigger."""

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
        self.dispatch_log = Path(self.temp_dir) / "logs" / "dispatch_audit.jsonl"
        self.agg_log = Path(self.temp_dir) / "logs" / "aggregation_events.jsonl"
        self.model = self._create_model()

        self.model_manager = GlobalModelManager(
            checkpoint_dir=self.checkpoint_dir,
            model=self.model,
            export_best=True,
        )
        self.dispatch_trigger = AutoDispatchTrigger(
            model_manager=self.model_manager,
            checkpoint_dir=self.checkpoint_dir,
            audit_log_file=self.dispatch_log,
            prefer_best=True,
        )
        self.sync_manager = RoundSyncManager(
            initial_round=1,
            min_clients_per_round=2,
        )
        self.strategy = FedMedStrategy(
            model_manager=self.model_manager,
            min_fit_clients=2,
            min_available_clients=2,
        )
        self.agg_trigger = AutoAggregateTrigger(
            sync_manager=self.sync_manager,
            strategy=self.strategy,
            model_manager=self.model_manager,
            min_upload_threshold=2,
            auto_advance_round=True,
            audit_log_file=self.agg_log,
        )

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # AutoDispatchTrigger Tests

    def test_dispatch_preset_baseline_when_no_checkpoints(self):
        """Test that baseline weights are automatically served if no training rounds occurred."""
        params, meta = self.dispatch_trigger.handle_client_connect(
            hospital_id="hospital_a",
            client_info={"os": "Windows", "gpu": "RTX 2050"},
        )

        self.assertIsNotNone(params)
        self.assertGreater(len(params), 0)
        self.assertEqual(meta["hospital_id"], "hospital_a")
        self.assertEqual(meta["model_version"], "preset_baseline_round_0")
        self.assertEqual(meta["round_num"], 0)

        # Verify audit log
        self.assertTrue(self.dispatch_log.exists())
        history = self.dispatch_trigger.get_dispatch_history()
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

        params, meta = self.dispatch_trigger.handle_client_connect(
            hospital_id="hospital_c",
            client_info={"dataset_size": 25},
        )

        self.assertEqual(meta["model_version"], "best_global_model")
        self.assertEqual(meta["round_num"], 2)
        self.assertAlmostEqual(meta["dice_score"], 0.85, places=4)
        self.assertIn("best_global_model.pth", meta["checkpoint_path"])

    def test_multiple_hospital_dispatches_and_stats(self):
        """Test multiple client requests and aggregate statistics tracking."""
        self.dispatch_trigger.handle_client_connect("hospital_a")
        self.dispatch_trigger.handle_client_connect("hospital_b")
        self.dispatch_trigger.handle_client_connect("hospital_a")

        stats = self.dispatch_trigger.get_dispatch_stats()
        self.assertEqual(stats["total_dispatches"], 3)
        self.assertEqual(stats["unique_hospitals_served"], 2)
        self.assertEqual(stats["hospitals_served_breakdown"]["hospital_a"], 2)
        self.assertEqual(stats["hospitals_served_breakdown"]["hospital_b"], 1)

    # AutoAggregateTrigger Tests

    def test_auto_aggregate_threshold_trigger(self):
        """Test that aggregation automatically executes when upload threshold is reached."""
        received_events = []

        def on_round_done(evt: RoundCompleteEvent):
            received_events.append(evt)

        self.agg_trigger.register_on_round_complete(on_round_done)

        base_params = get_model_parameters(self.model)
        p1 = [arr + 0.1 for arr in base_params]
        p2 = [arr + 0.2 for arr in base_params]

        # 1. Hospital A uploads (1/2 required) -> Should NOT trigger
        ok1, msg1, evt1 = self.agg_trigger.handle_client_upload(
            hospital_id="hospital_a",
            round_num=1,
            parameters=p1,
            num_examples=10,
            metrics={"train_loss": 0.5, "val_dice_mean": 0.70},
        )
        self.assertTrue(ok1)
        self.assertIsNone(evt1)
        self.assertEqual(len(received_events), 0)
        self.assertEqual(self.sync_manager.current_round, 1)

        # 2. Hospital B uploads (2/2 required) -> SHOULD trigger automated aggregation!
        ok2, msg2, evt2 = self.agg_trigger.handle_client_upload(
            hospital_id="hospital_b",
            round_num=1,
            parameters=p2,
            num_examples=15,
            metrics={"train_loss": 0.4, "val_dice_mean": 0.80},
        )
        self.assertTrue(ok2)
        self.assertIsNotNone(evt2)
        self.assertEqual(len(received_events), 1)

        # Verify RoundCompleteEvent payload
        self.assertEqual(evt2.round_num, 1)
        self.assertEqual(set(evt2.participating_clients), {"hospital_a", "hospital_b"})
        self.assertEqual(evt2.num_samples_total, 25)
        self.assertAlmostEqual(evt2.metrics["val_dice_mean"], 0.75, places=4)
        self.assertTrue(evt2.is_new_best)

        # Verify checkpoint file was generated
        self.assertTrue((self.checkpoint_dir / "global_model_round_001.pth").exists())
        self.assertTrue((self.checkpoint_dir / "best_global_model.pth").exists())

        # Verify round auto-advanced to 2
        self.assertEqual(self.sync_manager.current_round, 2)

        # Verify event log file
        self.assertTrue(self.agg_log.exists())
        history = self.agg_trigger.get_aggregation_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["round_num"], 1)

    def test_force_aggregate_trigger(self):
        """Test manual force aggregation on partial uploads."""
        base_params = get_model_parameters(self.model)
        self.agg_trigger.handle_client_upload(
            hospital_id="hospital_lone",
            round_num=1,
            parameters=base_params,
            num_examples=5,
            metrics={"train_loss": 0.6, "val_dice_mean": 0.50},
        )

        # Force aggregation with only 1 client
        triggered, event = self.agg_trigger.check_and_trigger(round_num=1, force=True)
        self.assertTrue(triggered)
        self.assertIsNotNone(event)
        self.assertEqual(event.participating_clients, ["hospital_lone"])
        self.assertTrue((self.checkpoint_dir / "global_model_round_001.pth").exists())


if __name__ == "__main__":
    unittest.main()
