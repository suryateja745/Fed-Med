"""
FedMed Central Backend Server - Unified CLI Entry Point.
Launches the FastAPI backend server using Uvicorn with full REST APIs,
WebSocket live telemetry streaming, authenticated AES-256-GCM model weight encryption,
and hospital node management.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.config import BackendConfig
from backend.app import create_app
from federation.utils.logger import setup_logger

try:
    import uvicorn
    HAS_UVICORN = True
except ImportError:
    HAS_UVICORN = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FedMed Production Backend Server Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind backend server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen for HTTP and WebSocket requests")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes (development)")
    parser.add_argument("--workers", type=int, default=1, help="Number of Uvicorn worker processes")
    parser.add_argument("--cors-origins", type=str, default="*", help="Comma-separated allowed CORS origins")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Path to model checkpoints")
    parser.add_argument("--logs-dir", type=str, default="./logs", help="Path to logs and dashboard JSON feeds")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to hospital private data folders")
    parser.add_argument("--dry-run", action="store_true", help="Execute backend self-test and route validation then exit")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(name="FedMedBackend")

    config = BackendConfig(
        host=args.host,
        port=args.port,
        cors_origins=args.cors_origins.split(",") if args.cors_origins != "*" else ["*"],
        checkpoint_dir=Path(args.checkpoint_dir),
        logs_dir=Path(args.logs_dir),
        data_dir=Path(args.data_dir),
    )

    app = create_app(config)

    logger.info("Initializing FedMed Production Backend...")
    logger.info(f"  * Host             : {config.host}:{config.port}")
    logger.info(f"  * Checkpoints      : {config.checkpoint_dir.resolve()}")
    logger.info(f"  * Logs & Telemetry : {config.logs_dir.resolve()}")
    logger.info(f"  * Interactive Docs : http://{config.host}:{config.port}/docs")
    logger.info(f"  * WebSocket Stream : ws://{config.host}:{config.port}/ws/telemetry")

    if args.dry_run:
        logger.info("[Dry Run] Executing backend test client self-test...")
        try:
            from fastapi.testclient import TestClient
            client = TestClient(app)

            # Test root
            r_root = client.get("/")
            assert r_root.status_code == 200, f"Root returned {r_root.status_code}"
            logger.info("  [✓] Root & metadata endpoint verified")

            # Test health
            r_health = client.get("/health")
            assert r_health.status_code == 200, f"Health returned {r_health.status_code}"
            logger.info("  [✓] Health check endpoint verified")

            # Test dashboard
            r_dash = client.get("/api/federation/dashboard")
            assert r_dash.status_code == 200, f"Dashboard returned {r_dash.status_code}"
            logger.info("  [✓] Federation dashboard endpoint verified")

            # Test security
            r_sec = client.get("/api/security/status")
            assert r_sec.status_code == 200, f"Security returned {r_sec.status_code}"
            logger.info(f"  [✓] Security endpoint verified (Key ID: {r_sec.json().get('global_weight_encryption', {}).get('key_id')})")

            # Test hospitals
            r_hosp = client.get("/api/hospitals")
            assert r_hosp.status_code == 200, f"Hospitals returned {r_hosp.status_code}"
            logger.info(f"  [✓] Hospital registry verified ({len(r_hosp.json())} nodes detected)")

            # Test database & JWT authentication
            r_login = client.post("/api/auth/login", json={"username": "admin", "password": "Admin@FedMed2026!"})
            assert r_login.status_code == 200, f"Auth login failed: {r_login.text}"
            token = r_login.json()["access_token"]
            logger.info("  [✓] Database & JWT login verified (Admin user authenticated)")

            # Test authenticated /me
            r_me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
            assert r_me.status_code == 200, f"Auth /me failed: {r_me.text}"
            logger.info(f"  [✓] RBAC Profile verified: {r_me.json()['username']} ({r_me.json()['role']})")

            # Test hospital node registry
            r_nodes = client.get("/api/auth/nodes")
            assert r_nodes.status_code == 200, f"Auth nodes failed: {r_nodes.text}"
            logger.info(f"  [✓] Database Hospital Nodes verified ({r_nodes.json()['total']} registered nodes)")

            logger.info("[Dry Run] All backend routes and database schemas validated successfully! (0 Errors)")
            return
        except Exception as e:
            logger.error(f"[Dry Run] Route validation failed: {e}")
            sys.exit(1)


    if not HAS_UVICORN:
        logger.error("Uvicorn is required to run the FedMed backend server. Please install uvicorn.")
        sys.exit(1)

    app_target = "backend.app:app" if (args.reload or args.workers > 1) else app
    uvicorn.run(
        app_target,
        host=config.host,
        port=config.port,
        reload=args.reload,
        workers=args.workers if not args.reload else None,
        log_level="info",
    )


if __name__ == "__main__":
    main()
