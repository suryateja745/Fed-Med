"""
Global Model Checkpoint and Weight Encryption Endpoints for FedMed.
Handles encrypted checkpoint streaming, HMAC-SHA256 signature verification,
model architecture inspection, and automated dispatch triggers.
"""

from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import FileResponse

from backend.dependencies import BackendServices, get_services
from backend.models import (
    CheckpointInfoResponse,
    ModelDispatchRequest,
    ModelDispatchResponse,
    ModelVerificationRequest,
    ModelVerificationResponse,
)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/global", response_model=Dict[str, Any])
def get_global_model_info(services: BackendServices = Depends(get_services)) -> Dict[str, Any]:
    """
    Return global model architecture specs, parameter counts, latest checkpoint information,
    and active encryption status.
    """
    ckpt_info = services.api_bridge.get_latest_checkpoint_info()
    metadata = services.model_manager.get_metadata()
    enc_status = services.model_manager.encryption_manager.get_security_status()

    return {
        "model_architecture": ckpt_info.get("model_architecture", "UNet3D"),
        "checkpoint_info": ckpt_info,
        "metadata_registry": metadata,
        "encryption": enc_status,
    }


@router.get("/download/latest")
def download_latest_checkpoint(
    encrypted: bool = Query(True, description="Download authenticated AES-256-GCM encrypted checkpoint (.pth.enc)"),
    services: BackendServices = Depends(get_services),
):
    """
    Download the latest global model checkpoint file.
    When encrypted=True, serves global_model_latest.pth.enc with cryptographic HMAC headers.
    """
    path = services.api_bridge.get_checkpoint_file(target="latest", encrypted=encrypted)
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Latest global model checkpoint file not found on server.",
        )

    # Calculate HMAC signature and verification headers
    headers = {
        "X-Model-Architecture": "UNet3D",
        "X-Model-Round": str(services.api_bridge.get_current_round()),
        "X-Model-Encrypted": "true" if path.name.endswith(".enc") else "false",
        "X-Model-Cipher": "AES-256-GCM" if path.name.endswith(".enc") else "NONE",
        "X-Model-Key-ID": services.encryption_manager.key_id,
    }

    if path.name.endswith(".enc"):
        verification = services.encryption_manager.verify_bundle_integrity(path)
        if verification.get("hmac_sha256"):
            headers["X-Model-HMAC"] = verification["hmac_sha256"]
        if verification.get("sha256"):
            headers["X-Model-SHA256"] = verification["sha256"]

    return FileResponse(
        path=path,
        filename=path.name,
        media_type="application/octet-stream",
        headers=headers,
    )


@router.get("/download/best")
def download_best_checkpoint(
    encrypted: bool = Query(True, description="Download authenticated AES-256-GCM encrypted checkpoint (.pth.enc)"),
    services: BackendServices = Depends(get_services),
):
    """
    Download the historical best-performing global model checkpoint based on validation Dice score.
    """
    path = services.api_bridge.get_checkpoint_file(target="best", encrypted=encrypted)
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Best global model checkpoint file not found on server.",
        )

    headers = {
        "X-Model-Architecture": "UNet3D",
        "X-Model-Round": str(services.model_manager.best_round),
        "X-Model-Best-Dice": str(services.model_manager.best_dice),
        "X-Model-Encrypted": "true" if path.name.endswith(".enc") else "false",
        "X-Model-Cipher": "AES-256-GCM" if path.name.endswith(".enc") else "NONE",
        "X-Model-Key-ID": services.encryption_manager.key_id,
    }

    if path.name.endswith(".enc"):
        verification = services.encryption_manager.verify_bundle_integrity(path)
        if verification.get("hmac_sha256"):
            headers["X-Model-HMAC"] = verification["hmac_sha256"]
        if verification.get("sha256"):
            headers["X-Model-SHA256"] = verification["sha256"]

    return FileResponse(
        path=path,
        filename=path.name,
        media_type="application/octet-stream",
        headers=headers,
    )


@router.get("/download/round/{round_num}")
def download_round_checkpoint(
    round_num: int,
    encrypted: bool = Query(True, description="Download authenticated encrypted checkpoint (.pth.enc)"),
    services: BackendServices = Depends(get_services),
):
    """
    Download global model checkpoint for a specific completed federated round.
    """
    path = services.api_bridge.get_checkpoint_file(target=str(round_num), encrypted=encrypted)
    if not path or not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Checkpoint for Round {round_num} not found on server.",
        )

    headers = {
        "X-Model-Architecture": "UNet3D",
        "X-Model-Round": str(round_num),
        "X-Model-Encrypted": "true" if path.name.endswith(".enc") else "false",
        "X-Model-Cipher": "AES-256-GCM" if path.name.endswith(".enc") else "NONE",
        "X-Model-Key-ID": services.encryption_manager.key_id,
    }

    return FileResponse(
        path=path,
        filename=path.name,
        media_type="application/octet-stream",
        headers=headers,
    )


@router.post("/verify", response_model=ModelVerificationResponse)
def verify_model_integrity(
    request: ModelVerificationRequest,
    services: BackendServices = Depends(get_services),
) -> ModelVerificationResponse:
    """
    Verify the cryptographic integrity and HMAC signature of an encrypted model checkpoint.
    Detects bit flips, tampering, model poisoning, or key mismatches.
    """
    target_path = request.file_path
    if not target_path:
        # Default to latest encrypted checkpoint
        latest_enc = services.config.checkpoint_dir / "global_model_latest.pth.enc"
        if latest_enc.exists():
            target_path = str(latest_enc)
        else:
            best_enc = services.config.checkpoint_dir / "best_global_model.pth.enc"
            if best_enc.exists():
                target_path = str(best_enc)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No file_path provided and no default encrypted checkpoints found.",
                )

    verification = services.encryption_manager.verify_bundle_integrity(target_path)
    return ModelVerificationResponse(**verification)


@router.post("/dispatch/{hospital_id}", response_model=ModelDispatchResponse)
def dispatch_model_to_hospital(
    hospital_id: str,
    payload: ModelDispatchRequest,
    services: BackendServices = Depends(get_services),
) -> ModelDispatchResponse:
    """
    Execute automated model dispatch trigger: resolves best/latest global model,
    encrypts parameters, and generates compliance audit log.
    """
    enc_payload, metadata = services.dispatch_trigger.get_encrypted_dispatch_payload(
        hospital_id=hospital_id,
        client_info=payload.client_info,
    )

    return ModelDispatchResponse(
        status="DISPATCHED",
        hospital_id=hospital_id,
        model_version=metadata.get("model_version", "best_global_model"),
        checkpoint_path=metadata.get("checkpoint_path", ""),
        round_num=metadata.get("round_num"),
        dice_score=metadata.get("dice_score"),
        encrypted=metadata.get("encrypted", True),
        cipher=metadata.get("cipher", "AES-256-GCM"),
        key_id=metadata.get("key_id"),
        hmac_sha256=metadata.get("hmac_sha256"),
        timestamp=metadata.get("timestamp", ""),
    )
