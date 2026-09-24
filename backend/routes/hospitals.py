"""
Hospital Client Node Management and Telemetry Routes for FedMed.
Handles registration, heartbeat keep-alives, hardware auditing, and local training history inspection.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import BackendServices, get_services
from backend.models import (
    HospitalHeartbeatRequest,
    HospitalHeartbeatResponse,
    HospitalNodeSummary,
    HospitalRegistrationRequest,
    HospitalRegistrationResponse,
)

router = APIRouter(prefix="/hospitals", tags=["Hospitals"])


@router.get("", response_model=List[Dict[str, Any]])
def list_hospitals(services: BackendServices = Depends(get_services)) -> List[Dict[str, Any]]:
    """
    List all registered and active hospital client nodes with connectivity status and local Dice scores.
    """
    return services.get_all_hospitals()


@router.get("/{hospital_id}", response_model=Dict[str, Any])
def get_hospital_details(
    hospital_id: str,
    services: BackendServices = Depends(get_services),
) -> Dict[str, Any]:
    """
    Retrieve full telemetry and local training history for a specific hospital node.
    """
    history = services.api_bridge.get_hospital_history(hospital_id)
    all_nodes = services.get_all_hospitals()
    node_info = next((n for n in all_nodes if n["hospital_id"] == hospital_id), None)

    return {
        "hospital_id": hospital_id,
        "node_info": node_info or {"status": "UNKNOWN"},
        "history": history,
    }


@router.post("/register", response_model=HospitalRegistrationResponse)
def register_hospital(
    request: HospitalRegistrationRequest,
    services: BackendServices = Depends(get_services),
) -> HospitalRegistrationResponse:
    """
    Register a local hospital client node.
    Generates node access token and provides active encryption key fingerprint.
    """
    reg = services.register_hospital(request.model_dump())
    return HospitalRegistrationResponse(
        status="REGISTERED",
        hospital_id=request.hospital_id,
        registered_at=reg["registered_at"],
        access_token=reg["access_token"],
        global_encryption_active=True,
        key_id=services.encryption_manager.key_id,
        message=f"Hospital node '{request.hospital_id}' registered successfully.",
    )


@router.post("/{hospital_id}/heartbeat", response_model=HospitalHeartbeatResponse)
def hospital_heartbeat(
    hospital_id: str,
    request: HospitalHeartbeatRequest,
    services: BackendServices = Depends(get_services),
) -> HospitalHeartbeatResponse:
    """
    Record keep-alive heartbeat ping from a hospital client node.
    Returns current global server round and directives.
    """
    hb = services.record_heartbeat(hospital_id, request.model_dump())
    server_round = services.api_bridge.get_current_round()

    return HospitalHeartbeatResponse(
        status="ACKNOWLEDGED",
        hospital_id=hospital_id,
        acknowledged_at=hb["timestamp"],
        server_round=server_round,
        global_model_ready=True,
        command="TRAIN" if request.status == "READY" else "CONTINUE",
    )


@router.get("/{hospital_id}/history", response_model=Dict[str, Any])
def get_hospital_history(
    hospital_id: str,
    services: BackendServices = Depends(get_services),
) -> Dict[str, Any]:
    """
    Fetch raw fit and evaluation telemetry history for a specific hospital node.
    """
    return services.api_bridge.get_hospital_history(hospital_id)
