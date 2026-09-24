"""
Unit test suite for FedMed Production Backend API.
Tests REST endpoints, model downloads, encryption status, hospital registration,
heartbeat telemetry, and control orchestration using FastAPI TestClient.
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from fastapi.testclient import TestClient

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.app import create_app
from backend.config import BackendConfig
from backend.dependencies import BackendServices


class TestBackendAPI(unittest.TestCase):
    """Test suite for FedMed Backend API endpoints."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.logs_dir = Path(self.temp_dir) / "logs"
        self.data_dir = Path(self.temp_dir) / "data"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.config = BackendConfig(
            checkpoint_dir=self.checkpoint_dir,
            logs_dir=self.logs_dir,
            data_dir=self.data_dir,
        )

        # Reset singleton instance for isolated test
        BackendServices._instance = None
        self.app = create_app(self.config)
        self.client = TestClient(self.app)

    def tearDown(self):
        BackendServices._instance = None
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_root_and_health_endpoints(self):
        """Test GET / and GET /health."""
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        data_root = res_root.json()
        self.assertEqual(data_root["status"], "ONLINE")
        self.assertIn("api_endpoints", data_root)

        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "HEALTHY")

    def test_federation_dashboard_and_status(self):
        """Test GET /api/federation/dashboard and GET /api/federation/status."""
        res_dash = self.client.get("/api/federation/dashboard")
        self.assertEqual(res_dash.status_code, 200)
        dash = res_dash.json()
        self.assertIn("federation_summary", dash)
        self.assertIn("checkpoint_info", dash)
        self.assertIn("security", dash)

        res_status = self.client.get("/api/federation/status")
        self.assertEqual(res_status.status_code, 200)
        status_data = res_status.json()
        self.assertEqual(status_data["current_round"], 0)
        self.assertIn("status", status_data)

    def test_hospital_registration_and_heartbeat(self):
        """Test POST /api/hospitals/register and POST /api/hospitals/{id}/heartbeat."""
        reg_payload = {
            "hospital_id": "hospital_alpha",
            "institution_name": "Alpha Medical Center",
            "gpu_name": "NVIDIA RTX 4090",
            "gpu_vram_gb": 24.0,
            "num_mri_scans": 150,
        }
        res_reg = self.client.post("/api/hospitals/register", json=reg_payload)
        self.assertEqual(res_reg.status_code, 200)
        reg_data = res_reg.json()
        self.assertEqual(reg_data["status"], "REGISTERED")
        self.assertEqual(reg_data["hospital_id"], "hospital_alpha")
        self.assertTrue(len(reg_data["access_token"]) > 16)

        # Heartbeat
        hb_payload = {
            "status": "TRAINING",
            "current_round": 1,
            "local_dice": 0.82,
            "vram_usage_gb": 4.5,
        }
        res_hb = self.client.post("/api/hospitals/hospital_alpha/heartbeat", json=hb_payload)
        self.assertEqual(res_hb.status_code, 200)
        hb_data = res_hb.json()
        self.assertEqual(hb_data["status"], "ACKNOWLEDGED")

        # Verify listed hospitals
        res_list = self.client.get("/api/hospitals")
        self.assertEqual(res_list.status_code, 200)
        hospitals = res_list.json()
        alpha = next((h for h in hospitals if h["hospital_id"] == "hospital_alpha"), None)
        self.assertIsNotNone(alpha)
        self.assertEqual(alpha["status"], "TRAINING")

    def test_security_and_key_rotation(self):
        """Test security telemetry, key status, and key rotation."""
        res_sec = self.client.get("/api/security/status")
        self.assertEqual(res_sec.status_code, 200)
        sec = res_sec.json()
        self.assertIn("global_weight_encryption", sec)
        self.assertEqual(sec["global_weight_encryption"]["cipher"], "AES-256-GCM")

        # Key status
        res_key = self.client.get("/api/security/keys/status")
        self.assertEqual(res_key.status_code, 200)
        key_data = res_key.json()
        old_id = key_data["key_id"]
        self.assertTrue(len(old_id) == 8)

        # Key rotation
        res_rot = self.client.post("/api/security/keys/rotate", json={"backup_existing": True})
        self.assertEqual(res_rot.status_code, 200)
        rot_data = res_rot.json()
        self.assertEqual(rot_data["status"], "ROTATED")
        self.assertEqual(rot_data["old_key_id"], old_id)
        self.assertNotEqual(rot_data["new_key_id"], old_id)

    def test_model_dispatch_and_verification(self):
        """Test automated model dispatch and verification."""
        # 1. Save dummy round checkpoint first to test verification
        services = BackendServices.get_instance()
        services.model_manager.save_round_checkpoint(
            round_num=1,
            metrics={"val_dice_mean": 0.85},
            participating_clients=["hospital_alpha"],
        )

        # 2. Test Model Dispatch
        res_disp = self.client.post(
            "/api/models/dispatch/hospital_alpha",
            json={"hospital_id": "hospital_alpha", "client_info": {"device": "cuda"}},
        )
        self.assertEqual(res_disp.status_code, 200)
        disp_data = res_disp.json()
        self.assertEqual(disp_data["status"], "DISPATCHED")
        self.assertTrue(disp_data["encrypted"])
        self.assertEqual(disp_data["cipher"], "AES-256-GCM")

        # 3. Test Model Verification endpoint
        res_ver = self.client.post("/api/models/verify", json={})
        self.assertEqual(res_ver.status_code, 200)
        ver_data = res_ver.json()
        self.assertTrue(ver_data["valid"])
        self.assertEqual(ver_data["status"], "VERIFIED")
        self.assertEqual(ver_data["cipher"], "AES-256-GCM")

        # 4. Test Model Download
        res_dl = self.client.get("/api/models/download/latest?encrypted=true")
        self.assertEqual(res_dl.status_code, 200)
        self.assertIn("X-Model-HMAC", res_dl.headers)
        self.assertEqual(res_dl.headers["X-Model-Cipher"], "AES-256-GCM")

    def test_control_endpoints(self):
        """Test training trigger, stop, and system health endpoints."""
        res_train = self.client.post(
            "/api/control/train/start",
            json={"num_rounds": 5, "min_clients": 2, "strategy": "FedMedStrategy"},
        )
        self.assertEqual(res_train.status_code, 200)
        train_data = res_train.json()
        self.assertEqual(train_data["status"], "TRAINING_INITIATED")
        self.assertEqual(train_data["target_rounds"], 5)

        res_stop = self.client.post("/api/control/train/stop")
        self.assertEqual(res_stop.status_code, 200)
        self.assertEqual(res_stop.json()["status"], "STOPPED")

        res_health = self.client.get("/api/control/system/health")
        self.assertEqual(res_health.status_code, 200)
        health_data = res_health.json()
        self.assertIn("cpu_cores", health_data)
        self.assertIn("cuda_available", health_data)


if __name__ == "__main__":
    unittest.main()
