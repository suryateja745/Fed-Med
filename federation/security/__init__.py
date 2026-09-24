"""Privacy-preserving ML, encryption (TenSEAL, AES-256-GCM), and Differential Privacy utilities."""

from federation.security.encryption import (
    EncryptedParameters,
    EncryptionContextManager,
    SecureClientHook,
    SecureServerHook,
    apply_differential_privacy,
    decrypt_parameters,
    encrypt_parameters,
    secure_aggregate_encrypted,
)
from federation.security.global_encryption import (
    CryptographicIntegrityError,
    EncryptionMetadata,
    GlobalKeyManager,
    GlobalWeightEncryptionManager,
    InvalidKeyError,
)

__all__ = [
    # Parameter & Homomorphic Encryption
    "EncryptedParameters",
    "EncryptionContextManager",
    "encrypt_parameters",
    "decrypt_parameters",
    "secure_aggregate_encrypted",
    "apply_differential_privacy",
    "SecureClientHook",
    "SecureServerHook",
    # Global Model Weight Encryption & Key Management
    "GlobalWeightEncryptionManager",
    "GlobalKeyManager",
    "EncryptionMetadata",
    "CryptographicIntegrityError",
    "InvalidKeyError",
]
