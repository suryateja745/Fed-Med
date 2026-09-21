"""
API Bridge and Structured Telemetry Adapters for FedMed.
Provides unified JSON feeds, state endpoints, and data adapters connecting the Flower federated
learning engine with Backend APIs (FastAPI/Flask) and Frontend dashboards (React/Streamlit).
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from federation.utils.config_loader import load_config
from federation.utils.logger import setup_logger


class FedMedAPIBridge:
    """
    Central API adapter bridge for FedMed.
    Provides structured methods to query real-time federated training metrics,
    participating hospital telemetry, model checkpoint registries, and system health.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "./checkpoints",
        logs_dir: Union[str, Path] = "./logs",
        data_dir: Union[str, Path] = "./data",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logs_dir = Path(logs_dir)
        self.data_dir = Path(data_dir)
        self.config = config or load_config()
        self.logger = setup_logger(name="FedMedAPIBridge")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def get_current_round(self) -> int:
        """Return the latest completed or active federated round number."""
        metadata = self._read_json(self.checkpoint_dir / "global_model_metadata.json")
        if metadata:
            return int(metadata.get("latest_round", 0))

        # Check aggregation events log
        events = self._read_jsonl(self.logs_dir / "aggregation_events.jsonl")
        if events:
            return int(events[-1].get("round_num", 0))

        return 0

    def get_latest_checkpoint_info(self) -> Dict[str, Any]:
        """Return detailed metadata about latest and best global model checkpoints."""
        metadata = self._read_json(self.checkpoint_dir / "global_model_metadata.json")
        best_path = self.checkpoint_dir / "best_global_model.pth"
        latest_path = self.checkpoint_dir / "global_model_latest.pth"

        best_size_mb = round(best_path.stat().st_size / (1024 * 1024), 2) if best_path.exists() else 0.0
        latest_size_mb = round(latest_path.stat().st_size / (1024 * 1024), 2) if latest_path.exists() else 0.0

        return {
            "latest_round": metadata.get("latest_round", 0) if metadata else 0,
            "best_round": metadata.get("best_round", 0) if metadata else 0,
            "best_dice_score": metadata.get("best_dice", -1.0) if metadata else -1.0,
            "has_best_checkpoint": best_path.exists(),
            "has_latest_checkpoint": latest_path.exists(),
            "best_checkpoint_path": str(best_path) if best_path.exists() else None,
            "latest_checkpoint_path": str(latest_path) if latest_path.exists() else None,
            "best_model_size_mb": best_size_mb,
            "latest_model_size_mb": latest_size_mb,
            "model_architecture": self.config.get("model", {}).get("name", "UNet3D"),
        }

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Return historical round-by-round convergence metrics."""
        metadata = self._read_json(self.checkpoint_dir / "global_model_metadata.json")
        if metadata and "rounds_history" in metadata:
            return metadata["rounds_history"]

        # Fallback to aggregation events log
        events = self._read_jsonl(self.logs_dir / "aggregation_events.jsonl")
        history = []
        for ev in events:
            history.append({
                "round": ev.get("round_num", 0),
                "timestamp": ev.get("timestamp"),
                "metrics": ev.get("metrics", {}),
                "global_dice_score": ev.get("metrics", {}).get("val_dice_mean"),
                "participating_clients": ev.get("participating_clients", []),
                "checkpoint_path": ev.get("checkpoint_path"),
                "is_best": ev.get("is_new_best", False),
            })
        return history

    def get_active_hospitals(self) -> List[Dict[str, Any]]:
        """Scan logs and data directories to discover participating hospital client nodes."""
        hospitals: Dict[str, Dict[str, Any]] = {}

        # 1. Check local hospital history JSON files in data/ and logs/
        search_dirs = [self.logs_dir, self.data_dir]
        for s_dir in search_dirs:
            if not s_dir.exists():
                continue
            for hist_file in s_dir.glob("*_history.json"):
                h_name = hist_file.stem.replace("_history", "").replace("sim_", "")
                data = self._read_json(hist_file)
                if data:
                    hospitals[h_name] = {
                        "hospital_id": h_name,
                        "status": "CONNECTED",
                        "latest_round": data.get("latest_round", data.get("current_round", 0)),
                        "best_local_dice": data.get("best_dice", data.get("best_local_dice", -1.0)),
                        "last_update": data.get("last_update", data.get("timestamp")),
                        "history_file": str(hist_file),
                    }

        # 2. Check dispatch audit events
        dispatch_events = self._read_jsonl(self.logs_dir / "dispatch_audit.jsonl")
        for ev in dispatch_events:
            h_name = ev.get("hospital_id")
            if h_name:
                if h_name not in hospitals:
                    hospitals[h_name] = {
                        "hospital_id": h_name,
                        "status": "IDLE",
                        "latest_round": ev.get("round_num", 0),
                        "best_local_dice": -1.0,
                        "last_update": ev.get("timestamp"),
                    }
                else:
                    hospitals[h_name]["last_update"] = ev.get("timestamp")

        # 3. Default known nodes if none detected yet
        if not hospitals:
            for default_hosp in ["hospital_a", "hospital_b", "hospital_c"]:
                hospitals[default_hosp] = {
                    "hospital_id": default_hosp,
                    "status": "READY",
                    "latest_round": 0,
                    "best_local_dice": -1.0,
                    "last_update": None,
                }

        return list(hospitals.values())

    def get_hospital_history(self, hospital_id: str) -> Dict[str, Any]:
        """Fetch local training and evaluation history for a specific hospital node."""
        candidates = [
            self.data_dir / hospital_id / f"{hospital_id}_history.json",
            self.logs_dir / f"{hospital_id}_history.json",
            self.logs_dir / f"sim_{hospital_id}_history.json",
        ]
        for path in candidates:
            if path.exists():
                data = self._read_json(path)
                if data:
                    return data

        return {"hospital_id": hospital_id, "rounds_history": [], "status": "NO_DATA"}

    def get_system_health(self) -> Dict[str, Any]:
        """Inspect server hardware, compute availability, and directory stats."""
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        gpu_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2) if cuda_avail else 0.0

        return {
            "status": "HEALTHY",
            "os": f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": cuda_avail,
            "gpu_count": torch.cuda.device_count() if cuda_avail else 0,
            "gpu_name": gpu_name,
            "gpu_vram_gb": gpu_vram,
            "cpu_cores": os.cpu_count() or 1,
            "checkpoint_count": len(list(self.checkpoint_dir.glob("*.pth"))),
            "log_files_count": len(list(self.logs_dir.glob("*.*"))),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_full_dashboard_state(self) -> Dict[str, Any]:
        """
        Consolidate all federated metrics, active hospitals, convergence history,
        and system health into a single unified JSON payload for frontend dashboards.
        """
        ckpt_info = self.get_latest_checkpoint_info()
        history = self.get_metrics_history()
        hospitals = self.get_active_hospitals()
        health = self.get_system_health()

        latest_metrics = history[-1].get("metrics", {}) if history else {}

        return {
            "project_name": "FedMed - Federated 3D Brain Tumor MRI Segmentation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "federation_summary": {
                "current_round": ckpt_info["latest_round"],
                "total_rounds_target": self.config.get("federation", {}).get("num_rounds", 10),
                "best_dice_score": ckpt_info["best_dice_score"],
                "best_round": ckpt_info["best_round"],
                "active_hospitals_count": len(hospitals),
                "min_clients_required": self.config.get("federation", {}).get("min_fit_clients", 2),
            },
            "latest_round_metrics": {
                "train_loss": latest_metrics.get("train_loss"),
                "val_loss": latest_metrics.get("val_loss"),
                "val_dice_mean": latest_metrics.get("val_dice_mean", ckpt_info["best_dice_score"]),
                "val_dice_tc": latest_metrics.get("val_dice_tc"),
                "val_dice_wt": latest_metrics.get("val_dice_wt"),
                "val_dice_et": latest_metrics.get("val_dice_et"),
            },
            "checkpoint_info": ckpt_info,
            "participating_hospitals": hospitals,
            "metrics_history": history,
            "system_health": health,
        }

    def export_dashboard_json(
        self,
        output_file: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Export current live dashboard state to JSON file on disk for polling by UI."""
        out_path = Path(output_file) if output_file is not None else (self.logs_dir / "fedmed_live_dashboard.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        state = self.get_full_dashboard_state()
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            self.logger.info(f"[APIBridge] Live dashboard state exported -> {out_path}")
        except Exception as e:
            self.logger.error(f"[APIBridge] Failed to export dashboard JSON: {e}")

        return out_path

    @staticmethod
    def _read_json(filepath: Path) -> Optional[Dict[str, Any]]:
        if not filepath.exists():
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _read_jsonl(filepath: Path) -> List[Dict[str, Any]]:
        events = []
        if not filepath.exists():
            return events
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception:
            pass
        return events
