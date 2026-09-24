"""
Pydantic Data Models and API Schemas for FedMed Backend.
Defines structured input/output contracts for Federation, Hospitals, Models, Security, and Control endpoints.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# 1. Federation & Telemetry Models

class FederationSummaryResponse(BaseModel):
    current_round: int = Field(..., description="Current or latest completed federated round")
    total_rounds_target: int = Field(..., description="Target federated training rounds")
    best_dice_score: float = Field(..., description="Historical highest global Dice score")
    best_round: int = Field(..., description="Round number that achieved the best Dice score")
    active_hospitals_count: int = Field(..., description="Number of currently active/connected hospitals")
    min_clients_required: int = Field(..., description="Minimum clients required per round")


class LatestRoundMetricsResponse(BaseModel):
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_dice_mean: Optional[float] = None
    val_dice_tc: Optional[float] = None
    val_dice_wt: Optional[float] = None
    val_dice_et: Optional[float] = None


class CheckpointInfoResponse(BaseModel):
    latest_round: int
    best_round: int
    best_dice_score: float
    has_best_checkpoint: bool
    has_latest_checkpoint: bool
    has_encrypted_best: bool = False
    has_encrypted_latest: bool = False
    best_checkpoint_path: Optional[str] = None
    latest_checkpoint_path: Optional[str] = None
    best_encrypted_path: Optional[str] = None
    latest_encrypted_path: Optional[str] = None
    best_model_size_mb: float = 0.0
    latest_model_size_mb: float = 0.0
    best_encrypted_size_mb: float = 0.0
    latest_encrypted_size_mb: float = 0.0
    model_architecture: str = "UNet3D"
    encryption: Dict[str, Any] = Field(default_factory=dict)


class HospitalNodeSummary(BaseModel):
    hospital_id: str
    status: str = "READY"
    latest_round: int = 0
    best_local_dice: float = -1.0
    last_update: Optional[str] = None
    history_file: Optional[str] = None


class DashboardStateResponse(BaseModel):
    project_name: str
    timestamp: str
    federation_summary: FederationSummaryResponse
    latest_round_metrics: LatestRoundMetricsResponse
    checkpoint_info: CheckpointInfoResponse
    participating_hospitals: List[HospitalNodeSummary]
    metrics_history: List[Dict[str, Any]]
    system_health: Dict[str, Any]
    security: Dict[str, Any]


# 2. Hospital Node Models

class HospitalRegistrationRequest(BaseModel):
    hospital_id: str = Field(..., description="Unique alphanumeric identifier (e.g. hospital_a)")
    institution_name: Optional[str] = Field("General Hospital", description="Full institutional name")
    contact_email: Optional[str] = Field(None, description="Technical contact email")
    gpu_name: Optional[str] = Field(None, description="Local GPU hardware name")
    gpu_vram_gb: Optional[float] = Field(None, description="Total GPU VRAM in GB")
    num_mri_scans: Optional[int] = Field(None, description="Total private local MRI scans available")
    public_key: Optional[str] = Field(None, description="Optional hospital public key for key exchange")


class HospitalRegistrationResponse(BaseModel):
    status: str = "REGISTERED"
    hospital_id: str
    registered_at: str
    access_token: str
    global_encryption_active: bool
    key_id: Optional[str] = None
    message: str


class HospitalHeartbeatRequest(BaseModel):
    status: str = Field("READY", description="Client status: READY, TRAINING, EVALUATING, IDLE, OFFLINE")
    current_round: Optional[int] = Field(None, description="Current round node is executing")
    local_dice: Optional[float] = Field(None, description="Most recent local validation Dice score")
    vram_usage_gb: Optional[float] = Field(None, description="Current GPU VRAM usage in GB")
    active_job_id: Optional[str] = None


class HospitalHeartbeatResponse(BaseModel):
    status: str = "ACKNOWLEDGED"
    hospital_id: str
    acknowledged_at: str
    server_round: int
    global_model_ready: bool = True
    command: Optional[str] = None


# 3. Model & Encryption Verification Models

class ModelVerificationRequest(BaseModel):
    file_path: Optional[str] = Field(None, description="Path to checkpoint file on server disk")
    expected_key_id: Optional[str] = Field(None, description="Expected key fingerprint ID")


class ModelVerificationResponse(BaseModel):
    valid: bool
    status: str
    error: Optional[str] = None
    key_id: Optional[str] = None
    cipher: Optional[str] = None
    round_num: Optional[int] = None
    dice_score: Optional[float] = None
    architecture: Optional[str] = None
    timestamp: Optional[str] = None
    sha256: Optional[str] = None
    hmac_sha256: Optional[str] = None
    bundle_size_bytes: Optional[int] = None


class ModelDispatchRequest(BaseModel):
    hospital_id: str
    client_info: Optional[Dict[str, Any]] = None


class ModelDispatchResponse(BaseModel):
    status: str
    hospital_id: str
    model_version: str
    checkpoint_path: str
    round_num: Optional[int]
    dice_score: Optional[float]
    encrypted: bool
    cipher: str
    key_id: Optional[str]
    hmac_sha256: Optional[str]
    timestamp: str


# 4. Security & Cryptographic Key Models

class SecurityStatusResponse(BaseModel):
    global_weight_encryption: Dict[str, Any]
    differential_privacy: Dict[str, Any]
    timestamp: str


class KeyRotationRequest(BaseModel):
    passphrase: Optional[str] = Field(None, description="Optional new passphrase to derive key")
    backup_existing: bool = Field(True, description="Backup prior key file before rotation")


class KeyRotationResponse(BaseModel):
    status: str
    old_key_id: str
    new_key_id: str
    timestamp: str


class ExportHospitalKeyResponse(BaseModel):
    status: str
    hospital_id: str
    key_id: str
    key_hex: str
    cipher: str = "AES-256-GCM"
    instructions: str


# 5. Control & Simulation Models

class TrainingStartRequest(BaseModel):
    num_rounds: Optional[int] = Field(None, description="Override total federated training rounds")
    min_clients: Optional[int] = Field(None, description="Override minimum clients required per round")
    strategy: Optional[str] = Field("FedMedStrategy", description="FedMedStrategy or FedAvg")


class TrainingStartResponse(BaseModel):
    status: str
    message: str
    target_rounds: int
    min_clients: int
    strategy: str
    timestamp: str


class SimulationTriggerRequest(BaseModel):
    num_clients: int = Field(3, description="Number of hospital clients to simulate (e.g. 3)")
    num_rounds: int = Field(3, description="Number of federated training rounds")
    partition_type: str = Field("quantity_skew", description="Partition distribution: quantity_skew, dirichlet, iid")


class SimulationTriggerResponse(BaseModel):
    status: str
    job_id: str
    message: str
    num_clients: int
    num_rounds: int
    partition_type: str
    timestamp: str
