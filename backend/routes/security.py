"""
Security and Cryptographic Key Management Endpoints for FedMed.
Handles global weight encryption telemetry, key status inspection, key rotation,
and authorized hospital decryption credential provisioning.
"""

from datetime import datetime, timezone
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.dependencies import BackendServices, get_services
from backend.models import (
    ExportHospitalKeyResponse,
    KeyRotationRequest,
    KeyRotationResponse,
    SecurityStatusResponse,
)

router = APIRouter(prefix="/security", tags=["Security"])


@router.get("/status", response_model=Dict[str, Any])
def get_security_status(services: BackendServices = Depends(get_services)) -> Dict[str, Any]:
    """
    Return comprehensive security posture:
    - Global model weight encryption (AES-256-GCM, key fingerprint, file count)
    - Differential Privacy calibration (epsilon, delta, clipping threshold)
    - Parameter homomorphic encryption status (TenSEAL CKKS)
    """
    return services.api_bridge.get_encryption_status()


@router.get("/keys/status", response_model=Dict[str, Any])
def get_key_status(services: BackendServices = Depends(get_services)) -> Dict[str, Any]:
    """
    Return active symmetric encryption key status, fingerprint ID, and storage path.
    Does NOT expose raw key bytes.
    """
    return services.encryption_manager.get_security_status()


@router.post("/keys/rotate", response_model=KeyRotationResponse)
def rotate_encryption_key(
    request: KeyRotationRequest,
    services: BackendServices = Depends(get_services),
) -> KeyRotationResponse:
    """
    Rotate the master global weight encryption key.
    Generates new 256-bit symmetric key, updates key manager, and automatically re-encrypts
    latest and best model checkpoints with the new key version.
    """
    try:
        new_key = None
        if request.passphrase:
            new_key = services.encryption_manager.key_manager.derive_key_from_passphrase(request.passphrase)

        result = services.model_manager.rotate_encryption_key(new_key=new_key)
        return KeyRotationResponse(
            status="ROTATED",
            old_key_id=result["old_key_id"],
            new_key_id=result["new_key_id"],
            timestamp=result["timestamp"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Key rotation failed: {e}",
        )


@router.post("/keys/export-hospital-key", response_model=ExportHospitalKeyResponse)
def export_hospital_key(
    hospital_id: str = Query(..., description="Hospital identifier e.g. hospital_a"),
    services: BackendServices = Depends(get_services),
) -> ExportHospitalKeyResponse:
    """
    Provision an authorized hospital client node with the active model decryption key.
    Allows local hospital nodes to decrypt inbound global checkpoints dispatched from the coordinator.
    """
    all_hospitals = services.get_all_hospitals()
    node = next((h for h in all_hospitals if h["hospital_id"] == hospital_id), None)
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hospital '{hospital_id}' is not registered with the FedMed coordinator.",
        )

    key_hex = services.encryption_manager.key.hex()
    key_id = services.encryption_manager.key_id

    return ExportHospitalKeyResponse(
        status="PROVISIONED",
        hospital_id=hospital_id,
        key_id=key_id,
        key_hex=key_hex,
        cipher="AES-256-GCM",
        instructions=(
            f"Use this 256-bit symmetric key to decrypt global model weights for {hospital_id}. "
            "Supply via --decrypt-key CLI argument when starting start_client.py."
        ),
    )
