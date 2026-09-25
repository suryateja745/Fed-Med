"""
FedMed Authentication and Role-Based Access Control (RBAC) Router.
Provides endpoints for:
1. User registration with institutional role segregation (POST /api/auth/register).
2. User and hospital login with JWT issuance (POST /api/auth/login).
3. Authenticated profile inspection (GET /api/auth/me).
4. Token refreshing (POST /api/auth/refresh).
5. Hospital node registry queries and provisioning (GET/POST /api/auth/nodes).
6. Compliance audit logs (GET /api/auth/audit-logs).
"""

from datetime import datetime, timezone
import json
import secrets
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.config import get_backend_config
from backend.database import (
    AuditLogEntry,
    HospitalNode,
    NodeStatus,
    User,
    UserRole,
    get_db,
)
from backend.models import (
    AuditLogResponse,
    AuthTokenResponse,
    HospitalNodeCreateRequest,
    TokenRefreshRequest,
    UserLoginRequest,
    UserProfileResponse,
    UserRegisterRequest,
)
from backend.security import (
    create_access_token,
    create_refresh_token,
    decode_jwt_token,
    get_current_user,
    hash_password,
    require_admin_role,
    require_auditor_role,
    verify_password,
)
from federation.utils.logger import setup_logger

logger = setup_logger(name="FedMedAuthRoutes")

auth_router = APIRouter(prefix="/auth", tags=["Authentication & RBAC"])


def _record_audit_log(
    db: Session,
    actor_id: str,
    actor_role: str,
    action: str,
    target_resource: str,
    client_ip: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Helper creating a persistent compliance audit log entry."""
    entry = AuditLogEntry(
        actor_id=actor_id,
        actor_role=actor_role,
        action=action,
        target_resource=target_resource,
        client_ip=client_ip,
        details=json.dumps(details or {}),
        is_tamper_flagged=False,
    )
    db.add(entry)
    db.flush()


@auth_router.post(
    "/register",
    response_model=AuthTokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new platform user or hospital staff account",
)
def register_user(
    req: UserRegisterRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Register a new user account with role segregation (ADMIN, HOSPITAL_STAFF, AUDITOR).
    Automatically hashes the password and generates unique user_id and JWT credentials.
    """
    # 1. Check existing username and email
    if db.query(User).filter(User.username == req.username.strip()).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{req.username}' is already taken",
        )
    if db.query(User).filter(User.email == req.email.strip().lower()).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Email address '{req.email}' is already registered",
        )

    # 2. Validate role
    role_str = req.role.upper().strip()
    try:
        user_role = UserRole(role_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{req.role}'. Allowed values: {[r.value for r in UserRole]}",
        )

    # 3. Generate unique user_id
    short_id = uuid.uuid4().hex[:6].upper()
    if user_role == UserRole.ADMIN:
        uid = f"USER-ADMIN-{short_id}"
    elif user_role == UserRole.AUDITOR:
        uid = f"USER-AUDIT-{short_id}"
    else:
        node_prefix = req.hospital_node_id.replace("NODE-", "") if req.hospital_node_id else "HOSP"
        uid = f"USER-{node_prefix}-{short_id}"

    # 4. Hash password
    p_hash, p_salt = hash_password(req.password)

    # 5. Create user
    new_user = User(
        user_id=uid,
        username=req.username.strip(),
        email=req.email.strip().lower(),
        password_hash=p_hash,
        salt=p_salt,
        role=user_role,
        institution_name=req.institution_name.strip(),
        hospital_node_id=req.hospital_node_id.strip() if req.hospital_node_id else None,
        is_active=True,
    )
    db.add(new_user)
    db.flush()

    # 6. Audit log
    client_ip = request.client.host if request.client else "unknown"
    _record_audit_log(
        db=db,
        actor_id=uid,
        actor_role=user_role.value,
        action="REGISTER_USER",
        target_resource=f"/api/auth/register:{req.username}",
        client_ip=client_ip,
        details={"institution": req.institution_name, "node_id": req.hospital_node_id},
    )

    # 7. Issue JWT tokens
    cfg = get_backend_config()
    token_claims = {
        "sub": new_user.username,
        "user_id": new_user.user_id,
        "role": user_role.value,
        "hospital_node_id": new_user.hospital_node_id,
        "institution_name": new_user.institution_name,
    }
    access_token = create_access_token(token_claims)
    refresh_token = create_refresh_token(token_claims)

    logger.info(f"Registered new user '{new_user.username}' as {user_role.value} ({uid})")

    return AuthTokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in_seconds=cfg.jwt_access_expire_minutes * 60,
        user=UserProfileResponse(**new_user.to_dict()),
    )


@auth_router.post(
    "/login",
    response_model=AuthTokenResponse,
    summary="Authenticate user and obtain JWT access and refresh tokens",
)
def login_user(
    req: UserLoginRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Authenticate against credentials for Admin, Hospital Staff, or Auditor accounts.
    Returns signed HS256 JWT access and refresh tokens.
    """
    user_query = req.username.strip()
    user = (
        db.query(User)
        .filter((User.username == user_query) | (User.email == user_query.lower()))
        .first()
    )

    if not user or not verify_password(req.password, user.password_hash, user.salt):
        logger.warning(f"Failed login attempt for identifier: '{user_query}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact central administrator.",
        )

    # Update login timestamp
    user.last_login_at = datetime.now(timezone.utc)
    db.flush()

    # Log audit event
    client_ip = request.client.host if request.client else "unknown"
    role_str = user.role.value if isinstance(user.role, UserRole) else str(user.role)
    _record_audit_log(
        db=db,
        actor_id=user.user_id,
        actor_role=role_str,
        action="LOGIN",
        target_resource=f"/api/auth/login:{user.username}",
        client_ip=client_ip,
    )

    # Generate tokens
    cfg = get_backend_config()
    token_claims = {
        "sub": user.username,
        "user_id": user.user_id,
        "role": role_str,
        "hospital_node_id": user.hospital_node_id,
        "institution_name": user.institution_name,
    }
    access_token = create_access_token(token_claims)
    refresh_token = create_refresh_token(token_claims)

    logger.info(f"User '{user.username}' ({role_str}) logged in successfully.")

    return AuthTokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in_seconds=cfg.jwt_access_expire_minutes * 60,
        user=UserProfileResponse(**user.to_dict()),
    )


@auth_router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get current authenticated user profile and roles",
)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """Returns profile, roles, and institutional affiliations of the current user."""
    return UserProfileResponse(**current_user.to_dict())


@auth_router.post(
    "/refresh",
    summary="Refresh an expired access token using a valid refresh token",
)
def refresh_token_endpoint(
    req: TokenRefreshRequest,
    db: Session = Depends(get_db),
):
    """Validates refresh token and returns a newly minted access token."""
    payload = decode_jwt_token(req.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provided token is not a refresh token",
        )

    username = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer active or registered",
        )

    cfg = get_backend_config()
    role_str = user.role.value if isinstance(user.role, UserRole) else str(user.role)
    new_claims = {
        "sub": user.username,
        "user_id": user.user_id,
        "role": role_str,
        "hospital_node_id": user.hospital_node_id,
        "institution_name": user.institution_name,
    }
    new_access_token = create_access_token(new_claims)

    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "expires_in_seconds": cfg.jwt_access_expire_minutes * 60,
    }


@auth_router.get(
    "/nodes",
    summary="List all registered institutional hospital client nodes",
)
def list_hospital_nodes(
    db: Session = Depends(get_db),
):
    """Returns the list of all registered hospital nodes with status and specs."""
    nodes = db.query(HospitalNode).order_by(HospitalNode.id.asc()).all()
    return {"hospital_nodes": [n.to_dict() for n in nodes], "total": len(nodes)}


@auth_router.post(
    "/nodes/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new hospital node with unique ID and hardware profile",
)
def register_hospital_node(
    req: HospitalNodeCreateRequest,
    request: Request,
    current_user: User = Depends(require_admin_role),
    db: Session = Depends(get_db),
):
    """Admin-only endpoint to provision a new institutional hospital node."""
    if db.query(HospitalNode).filter(HospitalNode.node_id == req.node_id.strip()).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Hospital node ID '{req.node_id}' already registered",
        )

    generated_api_key = f"fedmed_key_{req.node_id.lower()}_{secrets.token_hex(16)}"
    new_node = HospitalNode(
        node_id=req.node_id.strip(),
        hospital_name=req.hospital_name.strip(),
        region=req.region.strip(),
        api_key=generated_api_key,
        gpu_device=req.gpu_device.strip(),
        vram_gb=req.vram_gb,
        cpu_cores=req.cpu_cores,
        status=NodeStatus.ONLINE,
        dataset_path=req.dataset_path.strip(),
        local_sample_count=req.local_sample_count,
    )
    db.add(new_node)
    db.flush()

    client_ip = request.client.host if request.client else "unknown"
    _record_audit_log(
        db=db,
        actor_id=current_user.user_id,
        actor_role="ADMIN",
        action="REGISTER_NODE",
        target_resource=f"/api/auth/nodes:{req.node_id}",
        client_ip=client_ip,
        details={"hospital_name": req.hospital_name, "hardware": req.gpu_device},
    )

    logger.info(f"Registered new hospital node '{req.node_id}' ({req.hospital_name})")
    return new_node.to_dict()


@auth_router.get(
    "/audit-logs",
    response_model=List[AuditLogResponse],
    summary="Inspect regulatory compliance audit log trail (Auditor/Admin only)",
)
def get_audit_logs(
    limit: int = 50,
    current_user: User = Depends(require_auditor_role),
    db: Session = Depends(get_db),
):
    """Returns compliance and access audit logs for regulatory and governance inspection."""
    logs = (
        db.query(AuditLogEntry)
        .order_by(AuditLogEntry.timestamp.desc())
        .limit(min(limit, 200))
        .all()
    )
    return [AuditLogResponse(**log.to_dict()) for log in logs]
