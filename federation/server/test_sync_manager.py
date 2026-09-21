"""
Unit test suite for RoundSyncManager.
Tests thread-safe parameter ingestion, concurrency locks, race condition guards,
stale upload rejections, and round barrier synchronization.
"""

import sys
import threading
import time
import unittest
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.server.sync_manager import PendingClientUpdate, RoundSyncManager, StalePolicy


class TestRoundSyncManager(unittest.TestCase):
    """Test suite for server-side concurrency locks and round synchronization."""

    def setUp(self):
        self.sync_mgr = RoundSyncManager(
            initial_round=1,
            min_clients_per_round=3,
            stale_policy=StalePolicy.REJECT,
        )
        self.dummy_params = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        ]

    def test_single_upload_and_status(self):
        """Test single client upload and status telemetry reporting."""
        success, msg = self.sync_mgr.submit_update(
            hospital_id="hospital_a",
            round_num=1,
            parameters=self.dummy_params,
            num_examples=10,
            metrics={"train_loss": 0.5},
        )
        self.assertTrue(success)

        status = self.sync_mgr.get_sync_status()
        self.assertEqual(status["current_round"], 1)
        self.assertEqual(status["active_uploads_count"], 1)
        self.assertIn("hospital_a", status["active_hospitals"])
        self.assertFalse(status["is_round_ready"])

    def test_round_barrier_completion(self):
        """Test that round readiness triggers once min_clients_per_round is reached."""
        hospitals = ["hospital_a", "hospital_b", "hospital_c"]
        for hosp in hospitals:
            self.sync_mgr.submit_update(
                hospital_id=hosp,
                round_num=1,
                parameters=self.dummy_params,
                num_examples=10,
            )

        status = self.sync_mgr.get_sync_status()
        self.assertTrue(status["is_round_ready"])
        self.assertEqual(status["active_uploads_count"], 3)

        updates = self.sync_mgr.get_ready_updates()
        self.assertEqual(len(updates), 3)

    def test_stale_upload_rejection(self):
        """Test that uploads from previous rounds are rejected."""
        # Advance server to Round 2
        self.sync_mgr.advance_round(2)
        self.assertEqual(self.sync_mgr.current_round, 2)

        # Hospital A attempts to upload for Round 1
        success, msg = self.sync_mgr.submit_update(
            hospital_id="hospital_a",
            round_num=1,
            parameters=self.dummy_params,
            num_examples=10,
        )
        self.assertFalse(success)
        self.assertIn("stale", msg.lower())

        status = self.sync_mgr.get_sync_status()
        self.assertEqual(status["stale_rejections"], 1)
        self.assertEqual(status["active_uploads_count"], 0)

    def test_future_upload_buffering(self):
        """Test that uploads for future rounds are buffered and restored when round advances."""
        # Upload for Round 2 while server is on Round 1
        success, msg = self.sync_mgr.submit_update(
            hospital_id="hospital_future",
            round_num=2,
            parameters=self.dummy_params,
            num_examples=15,
        )
        self.assertTrue(success)
        self.assertIn("buffered", msg.lower())

        status = self.sync_mgr.get_sync_status()
        self.assertEqual(status["active_uploads_count"], 0)
        self.assertIn(2, status["future_buffered_rounds"])

        # Advance to Round 2 -> Buffered update should automatically appear in active queue
        self.sync_mgr.advance_round(2)
        status_r2 = self.sync_mgr.get_sync_status()
        self.assertEqual(status_r2["current_round"], 2)
        self.assertEqual(status_r2["active_uploads_count"], 1)
        self.assertIn("hospital_future", status_r2["active_hospitals"])

    def test_concurrent_simultaneous_uploads(self):
        """
        Test high-concurrency upload safety.
        Launches 20 simultaneous threads uploading to the sync manager.
        """
        sync = RoundSyncManager(initial_round=1, min_clients_per_round=10)
        num_threads = 20
        results = []

        def worker(client_idx: int):
            hosp_id = f"hospital_{client_idx:02d}"
            # Slightly vary parameters
            mod_params = [self.dummy_params[0] + client_idx, self.dummy_params[1]]
            ok, msg = sync.submit_update(
                hospital_id=hosp_id,
                round_num=1,
                parameters=mod_params,
                num_examples=10 + client_idx,
                metrics={"thread_id": client_idx},
            )
            results.append((hosp_id, ok, mod_params))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), num_threads)
        self.assertTrue(all(ok for _, ok, _ in results))

        status = sync.get_sync_status()
        self.assertEqual(status["active_uploads_count"], num_threads)
        self.assertTrue(status["is_round_ready"])

        # Verify no data corruption in parameter lists
        ready_updates = sync.get_ready_updates()
        self.assertEqual(len(ready_updates), num_threads)
        for update in ready_updates:
            cid = int(update.hospital_id.split("_")[1])
            expected_first = self.dummy_params[0] + cid
            np.testing.assert_allclose(update.parameters[0], expected_first)

    def test_wait_for_round_completion_timeout(self):
        """Test blocking wait_for_round_completion with timeout."""
        # Requires 3 clients, submit only 1
        self.sync_mgr.submit_update("hosp_1", 1, self.dummy_params, 5)

        start = time.time()
        ready, updates = self.sync_mgr.wait_for_round_completion(round_num=1, timeout=0.1)
        elapsed = time.time() - start

        self.assertFalse(ready)
        self.assertEqual(len(updates), 0)
        self.assertGreaterEqual(elapsed, 0.08)


if __name__ == "__main__":
    unittest.main()
