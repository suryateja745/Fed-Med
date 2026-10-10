"""
Unit test suite for FedMedAPIBridge.
Tests API state queries, dashboard JSON exports, metrics history aggregation,
checkpoint inspections, and hospital status tracking.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.api_bridge import FedMedAPIBridge


class TestFedMedAPIBridge(unittest.TestCase):
    """Test suite for API Bridge and telemetry exporters."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.logs_dir = Path(self.temp_dir) / "logs"
        self.data_dir = Path(self.temp_dir) / "data"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = FedMedAPIBridge(
            checkpoint_dir=self.checkpoint_dir,
            logs_dir=self.logs_dir,
            data_dir=self.data_dir,
        )

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_state_defaults(self):
        """Test that API Bridge returns valid default objects when no training has occurred."""
        self.assertEqual(self.bridge.get_current_round(), 0)

        ckpt_info = self.bridge.get_latest_checkpoint_info()
        self.assertEqual(ckpt_info["latest_round"], 0)
        self.assertFalse(ckpt_info["has_best_checkpoint"])

        history = self.bridge.get_metrics_history()
        self.assertEqual(len(history), 0)

        hospitals = self.bridge.get_active_hospitals()
        self.assertGreaterEqual(len(hospitals), 3)

        health = self.bridge.get_system_health()
        self.assertEqual(health["status"], "HEALTHY")
        self.assertIn("cpu_cores", health)

    def test_populated_state_queries(self):
        """Test queries against populated metadata and audit logs."""
        # 1. Write mock global_model_metadata.json
        metadata = {
            "project": "FedMed",
            "latest_round": 3,
            "best_round": 2,
            "best_dice": 0.885,
            "rounds_history": [
                {"round": 1, "global_dice_score": 0.65, "metrics": {"train_loss": 0.6}},
                {"round": 2, "global_dice_score": 0.885, "metrics": {"train_loss": 0.35}},
                {"round": 3, "global_dice_score": 0.82, "metrics": {"train_loss": 0.38}},
            ],
        }
        with open(self.checkpoint_dir / "global_model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        # 2. Touch mock checkpoints
        (self.checkpoint_dir / "best_global_model.pth").write_bytes(b"dummy_best")
        (self.checkpoint_dir / "global_model_latest.pth").write_bytes(b"dummy_latest")

        # 3. Write mock hospital client history
        hosp_a_data = {
            "hospital_id": "hospital_a",
            "latest_round": 3,
            "best_dice": 0.89,
            "timestamp": "2026-09-21T12:00:00Z",
        }
        with open(self.logs_dir / "hospital_a_history.json", "w", encoding="utf-8") as f:
            json.dump(hosp_a_data, f)

        # Verify API queries
        self.assertEqual(self.bridge.get_current_round(), 3)

        ckpt_info = self.bridge.get_latest_checkpoint_info()
        self.assertEqual(ckpt_info["latest_round"], 3)
        self.assertEqual(ckpt_info["best_round"], 2)
        self.assertAlmostEqual(ckpt_info["best_dice_score"], 0.885, places=3)
        self.assertTrue(ckpt_info["has_best_checkpoint"])
        self.assertTrue(ckpt_info["has_latest_checkpoint"])

        history = self.bridge.get_metrics_history()
        self.assertEqual(len(history), 3)
        self.assertAlmostEqual(history[1]["global_dice_score"], 0.885, places=3)

        hosp_history = self.bridge.get_hospital_history("hospital_a")
        self.assertEqual(hosp_history["hospital_id"], "hospital_a")
        self.assertAlmostEqual(hosp_history["best_dice"], 0.89, places=2)

    def test_export_dashboard_json(self):
        """Test exporting full dashboard state JSON to disk."""
        out_file = self.logs_dir / "test_dashboard.json"
        exported_path = self.bridge.export_dashboard_json(output_file=out_file)

        self.assertTrue(exported_path.exists())
        with open(exported_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("project_name", data)
        self.assertIn("federation_summary", data)
        self.assertIn("latest_round_metrics", data)
        self.assertIn("checkpoint_info", data)
        self.assertIn("participating_hospitals", data)
        self.assertIn("system_health", data)


if __name__ == "__main__":
    unittest.main()
