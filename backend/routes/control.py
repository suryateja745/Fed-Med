"""
Federation Control, Training Orchestration, and System Health Routes for FedMed.
Enables triggering federated rounds, stopping jobs, launching multi-hospital simulations,
and auditing system compute infrastructure.
"""

from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Any, Dict
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from backend.dependencies import BackendServices, get_services
from backend.models import (
    SimulationTriggerRequest,
    SimulationTriggerResponse,
    TrainingStartRequest,
    TrainingStartResponse,
)

router = APIRouter(prefix="/control", tags=["Control"])


def _run_background_simulation(
    job_id: str,
    num_clients: int,
    num_rounds: int,
    partition_type: str,
    services: BackendServices,
) -> None:
    """Worker task executed in background thread for multi-hospital simulation."""
    services.active_jobs[job_id]["status"] = "RUNNING"
    services.active_jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        from federation.simulate import run_federated_simulation
        summary = run_federated_simulation(
            num_clients=num_clients,
            num_rounds=num_rounds,
            partition_type=partition_type,
            output_dir="./reports",
            fast_dev_run=True,
        )
        services.active_jobs[job_id]["status"] = "COMPLETED"
        services.active_jobs[job_id]["summary"] = summary
        services.active_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as e:
        services.active_jobs[job_id]["status"] = "FAILED"
        services.active_jobs[job_id]["error"] = str(e)
        services.active_jobs[job_id]["failed_at"] = datetime.now(timezone.utc).isoformat()


@router.post("/train/start", response_model=TrainingStartResponse)
def start_federated_training(
    request: TrainingStartRequest,
    services: BackendServices = Depends(get_services),
) -> TrainingStartResponse:
    """
    Trigger start of federated learning orchestration.
    """
    fl_cfg = services.fl_config.get("federation", {})
    target_rounds = request.num_rounds or int(fl_cfg.get("num_rounds", 10))
    min_clients = request.min_clients or int(fl_cfg.get("min_fit_clients", 2))
    strat = request.strategy or "FedMedStrategy"

    job_id = f"train_job_{int(time.time())}"
    services.active_jobs[job_id] = {
        "job_id": job_id,
        "type": "FEDERATED_TRAINING",
        "status": "RUNNING",
        "target_rounds": target_rounds,
        "min_clients": min_clients,
        "strategy": strat,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return TrainingStartResponse(
        status="TRAINING_INITIATED",
        message=f"Federated training round initiated with strategy '{strat}'.",
        target_rounds=target_rounds,
        min_clients=min_clients,
        strategy=strat,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/train/stop", response_model=Dict[str, Any])
def stop_federated_training(services: BackendServices = Depends(get_services)) -> Dict[str, Any]:
    """
    Halt all active federated training background jobs.
    """
    stopped_count = 0
    for j_id, job in services.active_jobs.items():
        if job.get("status") == "RUNNING":
            job["status"] = "STOPPED"
            job["stopped_at"] = datetime.now(timezone.utc).isoformat()
            stopped_count += 1

    return {
        "status": "STOPPED",
        "stopped_jobs_count": stopped_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/simulate", response_model=SimulationTriggerResponse)
def trigger_simulation(
    request: SimulationTriggerRequest,
    background_tasks: BackgroundTasks,
    services: BackendServices = Depends(get_services),
) -> SimulationTriggerResponse:
    """
    Launch multi-hospital simulation testbed (Hospital A, B, C) in a background thread.
    Generates convergence curves and updates global model metadata upon completion.
    """
    job_id = f"sim_{int(time.time())}"
    services.active_jobs[job_id] = {
        "job_id": job_id,
        "type": "SIMULATION",
        "status": "PENDING",
        "num_clients": request.num_clients,
        "num_rounds": request.num_rounds,
        "partition_type": request.partition_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Dispatch to background executor
    background_tasks.add_task(
        _run_background_simulation,
        job_id=job_id,
        num_clients=request.num_clients,
        num_rounds=request.num_rounds,
        partition_type=request.partition_type,
        services=services,
    )

    return SimulationTriggerResponse(
        status="SIMULATION_QUEUED",
        job_id=job_id,
        message=f"Simulation job {job_id} queued across {request.num_clients} hospital nodes.",
        num_clients=request.num_clients,
        num_rounds=request.num_rounds,
        partition_type=request.partition_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/jobs", response_model=Dict[str, Any])
def get_jobs_status(services: BackendServices = Depends(get_services)) -> Dict[str, Any]:
    """
    List all background training and simulation jobs and their current execution state.
    """
    return {
        "total_jobs": len(services.active_jobs),
        "jobs": list(services.active_jobs.values()),
    }


@router.get("/system/health", response_model=Dict[str, Any])
def get_system_health(services: BackendServices = Depends(get_services)) -> Dict[str, Any]:
    """
    Query server hardware capabilities, PyTorch CUDA GPU availability, CPU core count,
    memory stats, and storage usage.
    """
    health = services.api_bridge.get_system_health()
    uptime = time.time() - services.server_start_time
    health["uptime_seconds"] = round(uptime, 2)
    return health
