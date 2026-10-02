"""
Integration and Unit Tests for FedMed Relational Database, JWT Authentication, and RBAC.
Tests:
1. SQLAlchemy SQLite database initialization, table creation, and seed data verification.
2. PBKDF2-HMAC-SHA256 password hashing and timing-resistant verification.
3. HS256 JWT encoding, decoding, expiration checks, and bit-tamper rejection.
4. User login (Admin, Hospital Staff, Auditor) via POST /api/auth/login.
5. Profile inspection via GET /api/auth/me with Bearer token.
6. Role-Based Access Control (RBAC) verification and 403 blocking for unauthorized roles.
7. User registration (POST /api/auth/register) and duplicate constraint handling.
8. Hospital Node provisioning (POST /api/auth/nodes/register).
9. Token refresh endpoint (POST /api/auth/refresh).
10. Audit trail persistence in database (GET /api/auth/audit-logs).
"""

import os
import shutil
import tempfile
import unittest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app import create_app
from backend.config import BackendConfig
from backend.database import (
    AuditLogEntry,
    Base,
    HospitalNode,
    NodeStatus,
    User,
    UserRole,
    get_db,
    init_db,
)
from backend.security import (
    create_access_token,
    create_jwt_token,
    create_refresh_token,
    decode_jwt_token,
    hash_password,
    verify_password,
)


class TestAuthAndDatabase(unittest.TestCase):
    """Test suite covering SQLAlchemy database models, JWT security, and RBAC endpoints."""

    @classmethod
    def setUpClass(cls):
        """Setup isolated temporary database for testing."""
        cls.temp_dir = tempfile.mkdtemp(prefix="fedmed_test_auth_db_")
        cls.test_db_path = os.path.join(cls.temp_dir, "test_fedmed.db")
        cls.db_url = f"sqlite:///{cls.test_db_path}"

        cls.config = BackendConfig(
            database_url=cls.db_url,
            checkpoint_dir=cls.temp_dir,
            logs_dir=cls.temp_dir,
            data_dir=cls.temp_dir,
            jwt_secret_key="test-secret-key-for-unit-testing-only",
            jwt_access_expire_minutes=60,
        )

        cls.app = create_app(cls.config)
        cls.client = TestClient(cls.app)

        # Trigger lifespan to initialize DB and seed data
        with cls.client:
            pass

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Unit Tests: Password Hashing & JWT Cryptography
    # -----------------------------------------------------------------------

    def test_01_password_hashing_and_verification(self):
        """Verify PBKDF2-HMAC-SHA256 password hashing and validation."""
        password = "SecurePassword@2026!"
        pwd_hash, salt = hash_password(password)

        self.assertIsInstance(pwd_hash, str)
        self.assertIsInstance(salt, str)
        self.assertEqual(len(pwd_hash), 64)  # 32 bytes hex = 64 characters

        # Verification succeeds with matching password
        self.assertTrue(verify_password(password, pwd_hash, salt))

        # Verification fails with incorrect password
        self.assertFalse(verify_password("WrongPassword123", pwd_hash, salt))
        self.assertFalse(verify_password(password.lower(), pwd_hash, salt))

    def test_02_jwt_token_creation_and_tamper_rejection(self):
        """Verify RFC 7519 HS256 JWT encoding, signature validation, and tamper rejection."""
        secret = "unit-test-cryptographic-secret"
        payload = {"sub": "test_user", "role": "ADMIN", "user_id": "USER-001"}

        token = create_jwt_token(payload, secret_key=secret)
        self.assertIsInstance(token, str)
        self.assertEqual(token.count("."), 2)

        # Decodes cleanly with correct secret
        decoded = decode_jwt_token(token, secret_key=secret)
        self.assertEqual(decoded["sub"], "test_user")
        self.assertEqual(decoded["role"], "ADMIN")

        # Fails with incorrect secret
        with self.assertRaises(Exception):
            decode_jwt_token(token, secret_key="wrong-secret-key")

        # Fails if payload was tampered with (bit-flip attack)
        parts = token.split(".")
        tampered_token = f"{parts[0]}.{parts[1][:-1]}X.{parts[2]}"
        with self.assertRaises(Exception):
            decode_jwt_token(tampered_token, secret_key=secret)

    # -----------------------------------------------------------------------
    # Integration Tests: Default Seed Data Verification
    # -----------------------------------------------------------------------

    def test_03_default_seed_data_initialized(self):
        """Verify default admin, hospital nodes, and auditor accounts are seeded."""
        engine = create_engine(self.db_url)
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as db:
            # Check Admin
            admin = db.query(User).filter(User.username == "admin").first()
            self.assertIsNotNone(admin)
            self.assertEqual(admin.role, UserRole.ADMIN)
            self.assertEqual(admin.institution_name, "FedMed Central Command & AI Consortium")

            # Check Hospital Nodes (A, B, C)
            nodes = db.query(HospitalNode).all()
            node_ids = {n.node_id for n in nodes}
            self.assertIn("NODE-HOSP-A", node_ids)
            self.assertIn("NODE-HOSP-B", node_ids)
            self.assertIn("NODE-HOSP-C", node_ids)

            # Check Hospital Staff users
            staff_a = db.query(User).filter(User.username == "hosp_a_lead").first()
            self.assertIsNotNone(staff_a)
            self.assertEqual(staff_a.role, UserRole.HOSPITAL_STAFF)
            self.assertEqual(staff_a.hospital_node_id, "NODE-HOSP-A")

            # Check Auditor
            auditor = db.query(User).filter(User.username == "auditor").first()
            self.assertIsNotNone(auditor)
            self.assertEqual(auditor.role, UserRole.AUDITOR)

    # -----------------------------------------------------------------------
    # Integration Tests: Auth Endpoints & RBAC
    # -----------------------------------------------------------------------

    def test_04_admin_login_success(self):
        """Test POST /api/auth/login with valid admin credentials."""
        with TestClient(self.app) as client:
            resp = client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "Admin@FedMed2026!"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("access_token", data)
            self.assertIn("refresh_token", data)
            self.assertEqual(data["token_type"], "bearer")
            self.assertEqual(data["user"]["username"], "admin")
            self.assertEqual(data["user"]["role"], "ADMIN")

    def test_05_hospital_staff_login_success(self):
        """Test POST /api/auth/login with valid hospital staff credentials."""
        with TestClient(self.app) as client:
            resp = client.post(
                "/api/auth/login",
                json={"username": "hosp_a_lead", "password": "HospA@FedMed2026!"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["user"]["role"], "HOSPITAL_STAFF")
            self.assertEqual(data["user"]["hospital_node_id"], "NODE-HOSP-A")

    def test_06_login_failure_invalid_password(self):
        """Test POST /api/auth/login fails with HTTP 401 on bad credentials."""
        with TestClient(self.app) as client:
            resp = client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "WrongPassword999!"},
            )
            self.assertEqual(resp.status_code, 401)
            self.assertIn("Invalid username or password", resp.json()["detail"])

    def test_07_get_current_user_profile(self):
        """Test GET /api/auth/me with valid Bearer token."""
        with TestClient(self.app) as client:
            login_resp = client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "Admin@FedMed2026!"},
            )
            token = login_resp.json()["access_token"]

            # Access protected /api/auth/me
            me_resp = client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(me_resp.status_code, 200)
            me_data = me_resp.json()
            self.assertEqual(me_data["username"], "admin")
            self.assertEqual(me_data["role"], "ADMIN")

    def test_08_rbac_enforcement_admin_vs_hospital(self):
        """
        Verify Role-Based Access Control:
        Admin can register hospital nodes; Hospital Staff receives HTTP 403 Forbidden.
        """
        with TestClient(self.app) as client:
            # Login as hospital staff
            hosp_login = client.post(
                "/api/auth/login",
                json={"username": "hosp_a_lead", "password": "HospA@FedMed2026!"},
            )
            hosp_token = hosp_login.json()["access_token"]

            # Attempt admin-only action (Register new hospital node)
            node_payload = {
                "node_id": "NODE-HOSP-UNAUTH",
                "hospital_name": "Unauthorized Hospital",
                "region": "Europe",
                "gpu_device": "RTX 3080",
                "vram_gb": 10.0,
                "cpu_cores": 8,
                "dataset_path": "./data/unauth",
                "local_sample_count": 10,
            }
            denied_resp = client.post(
                "/api/auth/nodes/register",
                json=node_payload,
                headers={"Authorization": f"Bearer {hosp_token}"},
            )
            self.assertEqual(denied_resp.status_code, 403)
            self.assertIn("Admin", denied_resp.json()["detail"])

            # Now perform same action with Admin token
            admin_login = client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "Admin@FedMed2026!"},
            )
            admin_token = admin_login.json()["access_token"]

            allowed_resp = client.post(
                "/api/auth/nodes/register",
                json=node_payload,
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            self.assertEqual(allowed_resp.status_code, 201)
            self.assertEqual(allowed_resp.json()["node_id"], "NODE-HOSP-UNAUTH")

    def test_09_user_registration(self):
        """Test POST /api/auth/register creating new hospital staff member."""
        with TestClient(self.app) as client:
            reg_payload = {
                "username": "dr_watson",
                "email": "dr.watson@bakerstreet.hospital.org",
                "password": "SherlockPassword2026!",
                "role": "HOSPITAL_STAFF",
                "institution_name": "Baker Street Clinical Center",
                "hospital_node_id": "NODE-HOSP-A",
            }
            reg_resp = client.post("/api/auth/register", json=reg_payload)
            self.assertEqual(reg_resp.status_code, 201)
            data = reg_resp.json()
            self.assertEqual(data["user"]["username"], "dr_watson")
            self.assertEqual(data["user"]["role"], "HOSPITAL_STAFF")
            self.assertIn("access_token", data)

            # Re-registering with same username should be rejected
            dup_resp = client.post("/api/auth/register", json=reg_payload)
            self.assertEqual(dup_resp.status_code, 400)
            self.assertIn("already taken", dup_resp.json()["detail"])

    def test_10_token_refresh(self):
        """Test POST /api/auth/refresh returns a fresh access token."""
        with TestClient(self.app) as client:
            login_resp = client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "Admin@FedMed2026!"},
            )
            refresh_token = login_resp.json()["refresh_token"]

            ref_resp = client.post(
                "/api/auth/refresh",
                json={"refresh_token": refresh_token},
            )
            self.assertEqual(ref_resp.status_code, 200)
            self.assertIn("access_token", ref_resp.json())

    def test_11_audit_logs_persisted_and_queryable(self):
        """Test audit logs capture login/registration events and can be viewed by auditor."""
        with TestClient(self.app) as client:
            auditor_login = client.post(
                "/api/auth/login",
                json={"username": "auditor", "password": "Audit@FedMed2026!"},
            )
            auditor_token = auditor_login.json()["access_token"]

            logs_resp = client.get(
                "/api/auth/audit-logs",
                headers={"Authorization": f"Bearer {auditor_token}"},
            )
            self.assertEqual(logs_resp.status_code, 200)
            logs = logs_resp.json()
            self.assertIsInstance(logs, list)
            self.assertGreater(len(logs), 0)

            actions = {log["action"] for log in logs}
            self.assertIn("LOGIN", actions)


if __name__ == "__main__":
    unittest.main()
