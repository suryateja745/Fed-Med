"""
Dependency Injection and Centralized Services Container for FedMed Backend.
Provides thread-safe access to the API Bridge, Global Model Manager, Encryption Engine,
Hospital Registry, and WebSocket Broadcaster.
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Set
import torch

from backend.config import BackendConfig, get_backend_config
from federation.api_bridge import FedMedAPIBridge
from federation.security.global_encryption import (
    GlobalKeyManager,
    GlobalWeightEncryptionManager,
)
from federation.server.model_manager import GlobalModelManager
from federation.server.triggers import AutoDispatchTrigger
from federation.utils.config_loader import load_config
from federation.utils.logger import setup_logger

try:
    from fastapi import WebSocket
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    WebSocket = Any


class WebSocketConnectionManager:
    """Manages active live WebSocket connections for streaming real-time telemetry to UI."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._lock = threading.Lock()
        self.logger = setup_logger(name="WebSocketManager")

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        with self._lock:
            self.active_connections.add(websocket)
        self.logger.info(f"WebSocket client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        with self._lock:
            self.active_connections.discard(websocket)
        self.logger.info(f"WebSocket client disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        dead_connections = []
        with self._lock:
            conns = list(self.active_connections)

        for conn in conns:
            try:
                await conn.send_json(message)
            except Exception:
                dead_connections.append(conn)

        if dead_connections:
            with self._lock:
                for dead in dead_connections:
                    self.active_connections.discard(dead)


class BackendServices:
    """
    Centralized Services Container for the FedMed Backend Application.
    Instantiates and manages core coordinators in a thread-safe singleton pattern.
    """

    _instance: Optional["BackendServices"] = None
    _init_lock = threading.Lock()

    def __init__(self, config: Optional[BackendConfig] = None) -> None:
        self.config = config or get_backend_config()
        self.logger = setup_logger(name="BackendServices")
        self.fl_config = load_config()

        # 1. Encryption Manager
        key_path = self.config.checkpoint_dir / "fedmed_global_model.key"
        self.key_manager = GlobalKeyManager(key_path=key_path, auto_generate=True)
        self.encryption_manager = GlobalWeightEncryptionManager(key_manager=self.key_manager)

        # 2. Global Model Manager (with Encryption enabled)
        self.model_manager = GlobalModelManager(
            checkpoint_dir=self.config.checkpoint_dir,
            config=self.fl_config,
            export_best=True,
            enable_encryption=True,
            encryption_manager=self.encryption_manager,
        )

        # 3. API Bridge
        self.api_bridge = FedMedAPIBridge(
            checkpoint_dir=self.config.checkpoint_dir,
            logs_dir=self.config.logs_dir,
            data_dir=self.config.data_dir,
            config=self.fl_config,
        )

        # 4. Dispatch Trigger
        self.dispatch_trigger = AutoDispatchTrigger(
            model_manager=self.model_manager,
            checkpoint_dir=self.config.checkpoint_dir,
            audit_log_file=self.config.logs_dir / "dispatch_audit.jsonl",
        )

        # 5. Live WebSocket Manager
        self.ws_manager = WebSocketConnectionManager()

        # 6. Registered Hospitals Dynamic Cache
        self._hospitals_lock = threading.Lock()
        self._registered_hospitals: Dict[str, Dict[str, Any]] = {}
        self._hospital_heartbeats: Dict[str, Dict[str, Any]] = {}
        self._hospital_registry_file = self.config.logs_dir / "hospital_registry.json"
        self._load_hospital_registry()

        # 7. Background Job Tracking
        self.active_jobs: Dict[str, Dict[str, Any]] = {}

        self.server_start_time = time.time()
        self.logger.info("Initialized FedMed Backend Services successfully.")

    @classmethod
    def get_instance(cls, config: Optional[BackendConfig] = None) -> "BackendServices":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    def _load_hospital_registry(self) -> None:
        """Load persistent registered hospital nodes from JSON file."""
        if self._hospital_registry_file.exists():
            try:
                with open(self._hospital_registry_file, "r", encoding="utf-8") as f:
                    self._registered_hospitals = json.load(f)
            except Exception:
                self._registered_hospitals = {}

    def _save_hospital_registry(self) -> None:
        """Save persistent registered hospital nodes to JSON file."""
        try:
            with open(self._hospital_registry_file, "w", encoding="utf-8") as f:
                json.dump(self._registered_hospitals, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save hospital registry: {e}")

    def register_hospital(self, req_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register or update a hospital node."""
        h_id = req_data["hospital_id"]
        token_seed = f"{h_id}_{datetime.now(timezone.utc).isoformat()}_{self.encryption_manager.key_id}"
        token = hashlib.sha256(token_seed.encode("utf-8")).hexdigest()

        record = {
            "hospital_id": h_id,
            "institution_name": req_data.get("institution_name", "Medical Center"),
            "contact_email": req_data.get("contact_email"),
            "gpu_name": req_data.get("gpu_name"),
            "gpu_vram_gb": req_data.get("gpu_vram_gb"),
            "num_mri_scans": req_data.get("num_mri_scans"),
            "public_key": req_data.get("public_key"),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "access_token": token,
            "status": "READY",
        }

        with self._hospitals_lock:
            self._registered_hospitals[h_id] = record
            self._save_hospital_registry()

        self.logger.info(f"Registered hospital node: {h_id} ({record['institution_name']})")
        return record

    def record_heartbeat(self, h_id: str, hb_data: Dict[str, Any]) -> Dict[str, Any]:
        """Record a heartbeat ping from a hospital node."""
        now_iso = datetime.now(timezone.utc).isoformat()
        heartbeat_entry = {
            "hospital_id": h_id,
            "timestamp": now_iso,
            **hb_data,
        }

        with self._hospitals_lock:
            self._hospital_heartbeats[h_id] = heartbeat_entry
            if h_id in self._registered_hospitals:
                self._registered_hospitals[h_id]["status"] = hb_data.get("status", "CONNECTED")
                self._registered_hospitals[h_id]["last_heartbeat"] = now_iso
                if "local_dice" in hb_data and hb_data["local_dice"] is not None:
                    self._registered_hospitals[h_id]["latest_local_dice"] = hb_data["local_dice"]
                if "current_round" in hb_data and hb_data["current_round"] is not None:
                    self._registered_hospitals[h_id]["current_round"] = hb_data["current_round"]

        return heartbeat_entry

    def get_all_hospitals(self) -> List[Dict[str, Any]]:
        """Consolidate registered hospitals and filesystem/log-discovered nodes."""
        discovered = self.api_bridge.get_active_hospitals()
        hospitals_map: Dict[str, Dict[str, Any]] = {h["hospital_id"]: h for h in discovered}

        with self._hospitals_lock:
            for h_id, reg in self._registered_hospitals.items():
                if h_id in hospitals_map:
                    hospitals_map[h_id].update({
                        "institution_name": reg.get("institution_name"),
                        "gpu_name": reg.get("gpu_name"),
                        "gpu_vram_gb": reg.get("gpu_vram_gb"),
                        "status": reg.get("status", hospitals_map[h_id].get("status")),
                        "registered": True,
                    })
                else:
                    hospitals_map[h_id] = {
                        "hospital_id": h_id,
                        "institution_name": reg.get("institution_name"),
                        "status": reg.get("status", "READY"),
                        "latest_round": reg.get("current_round", 0),
                        "best_local_dice": reg.get("latest_local_dice", -1.0),
                        "last_update": reg.get("last_heartbeat", reg.get("registered_at")),
                        "gpu_name": reg.get("gpu_name"),
                        "gpu_vram_gb": reg.get("gpu_vram_gb"),
                        "registered": True,
                    }

        return list(hospitals_map.values())


def get_services() -> BackendServices:
    """FastAPI Dependency Provider for BackendServices."""
    return BackendServices.get_instance()
