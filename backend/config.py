"""
Backend Configuration and Environment Management for FedMed.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List


@dataclass
class BackendConfig:
    """Backend server configuration settings with environment variable overrides."""
    host: str = os.getenv("FEDMED_BACKEND_HOST", "0.0.0.0")
    port: int = int(os.getenv("FEDMED_BACKEND_PORT", "8000"))
    cors_origins: List[str] = field(
        default_factory=lambda: os.getenv("FEDMED_CORS_ORIGINS", "*").split(",")
    )
    checkpoint_dir: Path = Path(os.getenv("FEDMED_CHECKPOINT_DIR", "./checkpoints"))
    logs_dir: Path = Path(os.getenv("FEDMED_LOGS_DIR", "./logs"))
    data_dir: Path = Path(os.getenv("FEDMED_DATA_DIR", "./data"))
    fl_server_address: str = os.getenv("FEDMED_FL_SERVER_ADDRESS", "127.0.0.1:8080")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/fedmed.db")
    jwt_secret_key: str = os.getenv("FEDMED_JWT_SECRET_KEY", "fedmed-super-secret-jwt-key-for-medical-ai-2026")
    jwt_algorithm: str = os.getenv("FEDMED_JWT_ALGORITHM", "HS256")
    jwt_access_expire_minutes: int = int(os.getenv("FEDMED_JWT_ACCESS_EXPIRE_MIN", "1440"))  # 24 hours
    jwt_refresh_expire_days: int = int(os.getenv("FEDMED_JWT_REFRESH_EXPIRE_DAYS", "30"))    # 30 days
    api_prefix: str = "/api"
    title: str = "FedMed Federated Learning Backend API"
    version: str = "1.0.0"
    description: str = (
        "Production-grade Backend for FedMed: 3D Brain Tumor MRI Federated Segmentation. "
        "Supports authenticated AES-256-GCM model weight encryption, real-time telemetry streaming, "
        "relational database persistence (SQLite/MySQL), role-based authentication, and automated triggers."
    )


def get_backend_config() -> BackendConfig:
    """Instantiate and return backend configuration."""
    cfg = BackendConfig()
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg

