"""Privacy-preserving ML, encryption (TenSEAL), and Differential Privacy utilities."""

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

__all__ = [
    "EncryptedParameters",
    "EncryptionContextManager",
    "encrypt_parameters",
    "decrypt_parameters",
    "secure_aggregate_encrypted",
    "apply_differential_privacy",
    "SecureClientHook",
    "SecureServerHook",
]

