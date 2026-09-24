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
    api_prefix: str = "/api"
    title: str = "FedMed Federated Learning Backend API"
    version: str = "1.0.0"
    description: str = (
        "Production-grade Backend for FedMed: 3D Brain Tumor MRI Federated Segmentation. "
        "Supports authenticated AES-256-GCM model weight encryption, real-time telemetry streaming, "
        "hospital node management, and automated trigger orchestration."
    )


def get_backend_config() -> BackendConfig:
    """Instantiate and return backend configuration."""
    cfg = BackendConfig()
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg
