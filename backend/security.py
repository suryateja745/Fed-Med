"""
FedMed Security and Cryptographic Authentication Utilities.
Implements:
1. PBKDF2-HMAC-SHA256 password hashing with random salt and timing-attack resistance.
2. Standard RFC 7519 HS256 JWT token generation, signature validation, and payload extraction.
3. Role-Based Access Control (RBAC) FastAPI dependencies for Admin, Hospital, and Auditor roles.
"""

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
import secrets
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.config import get_backend_config
from backend.database import User, UserRole, get_db
from federation.utils.logger import setup_logger

logger = setup_logger(name="FedMedAuth")

# HTTP Bearer authentication scheme
security_bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Password Hashing & Verification (PBKDF2-HMAC-SHA256)
# ---------------------------------------------------------------------------

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with 100,000 iterations.
    Returns (hex_hash, hex_salt).
    """
    if salt is None:
        salt_bytes = secrets.token_bytes(16)
        salt = salt_bytes.hex()
    else:
        salt_bytes = bytes.fromhex(salt)

    iterations = 100_000
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        iterations,
        dklen=32,
    )
    return derived.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """
    Verifies a plain password against the stored hex hash and salt.
    Uses constant-time comparison to prevent timing attacks.
    """
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, password_hash)


# ---------------------------------------------------------------------------
# RFC 7519 HS256 JWT Token Implementation
# ---------------------------------------------------------------------------

def _b64url_encode(data: bytes) -> str:
    """Base64url encode without trailing '=' padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    """Base64url decode with padding restoration."""
    padding = len(data) % 4
    if padding:
        data += "=" * (4 - padding)
    return base64.urlsafe_b64decode(data.encode("ascii"))


def create_jwt_token(payload: Dict[str, Any], secret_key: Optional[str] = None) -> str:
    """
    Encodes and cryptographically signs a JWT token using HMAC-SHA256 (HS256).
    """
    cfg = get_backend_config()
    secret = (secret_key or cfg.jwt_secret_key).encode("utf-8")

    header = {"alg": "HS256", "typ": "JWT"}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_json = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")

    header_b64 = _b64url_encode(header_json)
    payload_b64 = _b64url_encode(payload_json)

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(secret, signing_input, hashlib.sha256).digest()
    signature_b64 = _b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_jwt_token(token: str, secret_key: Optional[str] = None, verify_exp: bool = True) -> Dict[str, Any]:
    """
    Validates signature and expiration of an HS256 JWT token, returning the payload dictionary.
    Raises HTTPException(401) on invalid signature, malformed token, or expiration.
    """
    cfg = get_backend_config()
    secret = (secret_key or cfg.jwt_secret_key).encode("utf-8")

    parts = token.strip().split(".")
    if len(parts) != 3:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed JWT token structure",
            headers={"WWW-Authenticate": "Bearer"},
        )

    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

    # Verify signature
    expected_sig = hmac.new(secret, signing_input, hashlib.sha256).digest()
    try:
        actual_sig = _b64url_decode(signature_b64)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid JWT signature encoding",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not hmac.compare_digest(actual_sig, expected_sig):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Cryptographic JWT signature verification failed",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Parse payload
    try:
        payload_bytes = _b64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid JWT payload: {str(exc)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check expiration
    if verify_exp and "exp" in payload:
        now_ts = datetime.now(timezone.utc).timestamp()
        if now_ts > float(payload["exp"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="JWT token has expired. Please re-authenticate.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return payload


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generate a signed access JWT token."""
    cfg = get_backend_config()
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=cfg.jwt_access_expire_minutes))
    to_encode.update({
        "exp": expire.timestamp(),
        "iat": now.timestamp(),
        "type": "access",
    })
    return create_jwt_token(to_encode)


def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generate a signed refresh JWT token."""
    cfg = get_backend_config()
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(days=cfg.jwt_refresh_expire_days))
    to_encode.update({
        "exp": expire.timestamp(),
        "iat": now.timestamp(),
        "type": "refresh",
    })
    return create_jwt_token(to_encode)


# ---------------------------------------------------------------------------
# FastAPI Dependency & Role-Based Access Control (RBAC)
# ---------------------------------------------------------------------------

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency extracting and validating the Bearer JWT token from the Authorization header.
    Returns the authenticated User model.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Bearer authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_jwt_token(token)

    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject identifier (sub)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"User '{username}' no longer exists in system registry",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated. Contact platform coordinator.",
        )

    return user


async def require_admin_role(current_user: User = Depends(get_current_user)) -> User:
    """Enforces that the authenticated user possesses the ADMIN / Coordinator role."""
    user_role = current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role)
    if user_role != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Central Coordinator Administrator privileges required for this action.",
        )
    return current_user


async def require_hospital_role(current_user: User = Depends(get_current_user)) -> User:
    """Enforces that the user is authorized for Hospital Node operations (HOSPITAL_STAFF or ADMIN)."""
    user_role = current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role)
    if user_role not in (UserRole.HOSPITAL_STAFF.value, UserRole.ADMIN.value):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hospital Node clinical staff privileges required for this action.",
        )
    return current_user


async def require_auditor_role(current_user: User = Depends(get_current_user)) -> User:
    """Enforces that the user is authorized for Regulatory/Audit review (AUDITOR or ADMIN)."""
    user_role = current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role)
    if user_role not in (UserRole.AUDITOR.value, UserRole.ADMIN.value):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clinical Governance & Audit privileges required for this action.",
        )
    return current_user
