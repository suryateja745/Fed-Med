"""
Federation Telemetry and Dashboard API Routes for FedMed.
Provides endpoints for querying live round status, convergence curves, and dashboard state.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends

from backend.dependencies import BackendServices, get_services
from backend.models import DashboardStateResponse, FederationSummaryResponse

router = APIRouter(prefix="/federation", tags=["Federation"])


@router.get("/dashboard", response_model=Dict[str, Any])
def get_dashboard(services: BackendServices = Depends(get_services)) -> Dict[str, Any]:
    """
    Return full live dashboard state for UI polling.
    Consolidates round progress, metrics, hospital nodes, system health, and security telemetry.
    """
    state = services.api_bridge.get_full_dashboard_state()
    # Augment with dynamic registered hospital list
    state["participating_hospitals"] = services.get_all_hospitals()
    state["federation_summary"]["active_hospitals_count"] = len(state["participating_hospitals"])
    return state


@router.get("/status", response_model=Dict[str, Any])
def get_status(services: BackendServices = Depends(get_services)) -> Dict[str, Any]:
    """
    Return real-time federation status, current round, and node connectivity.
    """
    ckpt_info = services.api_bridge.get_latest_checkpoint_info()
    hospitals = services.get_all_hospitals()
    training_jobs = [j for j in services.active_jobs.values() if j.get("status") == "RUNNING"]

    return {
        "status": "TRAINING" if training_jobs else "IDLE",
        "current_round": ckpt_info["latest_round"],
        "target_rounds": services.fl_config.get("federation", {}).get("num_rounds", 10),
        "best_round": ckpt_info["best_round"],
        "best_dice_score": ckpt_info["best_dice_score"],
        "active_hospitals_count": len(hospitals),
        "min_clients_required": services.fl_config.get("federation", {}).get("min_fit_clients", 2),
        "active_training_jobs": len(training_jobs),
    }


@router.get("/metrics", response_model=List[Dict[str, Any]])
def get_metrics_history(services: BackendServices = Depends(get_services)) -> List[Dict[str, Any]]:
    """
    Return round-by-round historical training and validation convergence metrics.
    Includes train loss, val loss, and sub-region Dice scores (Mean, TC, WT, ET).
    """
    return services.api_bridge.get_metrics_history()


@router.get("/summary", response_model=Dict[str, Any])
def get_federation_summary(services: BackendServices = Depends(get_services)) -> Dict[str, Any]:
    """
    Return high-level summary of the FedMed project and current model version.
    """
    ckpt_info = services.api_bridge.get_latest_checkpoint_info()
    return {
        "project": "FedMed - 3D Brain Tumor MRI Segmentation",
        "framework": "Flower + PyTorch + MONAI",
        "model_architecture": ckpt_info.get("model_architecture", "UNet3D"),
        "latest_round": ckpt_info["latest_round"],
        "best_dice_score": ckpt_info["best_dice_score"],
        "encryption": ckpt_info.get("encryption", {}),
    }
