from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import tenseal as ts


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PUBLIC_CONTEXT_FILE = DATA_DIR / "tenseal_public_context.bin"
SECRET_CONTEXT_FILE = DATA_DIR / "tenseal_secret_context.bin"

POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
GLOBAL_SCALE = 2**40


def create_context() -> ts.Context:
    """Create the private TenSEAL CKKS context."""

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    context.global_scale = GLOBAL_SCALE
    return context


def ensure_context_files() -> None:
    """Create private and public context files once."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if SECRET_CONTEXT_FILE.exists() and PUBLIC_CONTEXT_FILE.exists():
        return

    context = create_context()

    SECRET_CONTEXT_FILE.write_bytes(
        context.serialize(
            save_public_key=True,
            save_secret_key=True,
            save_galois_keys=False,
            save_relin_keys=False,
        )
    )

    context.make_context_public()

    PUBLIC_CONTEXT_FILE.write_bytes(
        context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=False,
            save_relin_keys=False,
        )
    )


def load_private_context() -> ts.Context:
    ensure_context_files()

    return ts.context_from(
        SECRET_CONTEXT_FILE.read_bytes()
    )


def load_public_context() -> ts.Context:
    ensure_context_files()

    return ts.context_from(
        PUBLIC_CONTEXT_FILE.read_bytes()
    )


def encrypt_update(values: np.ndarray | Iterable[float]) -> bytes:
    """Encrypt a local model update and return serialized ciphertext."""

    public_context = load_public_context()

    array = np.asarray(values, dtype=np.float64).reshape(-1)

    encrypted = ts.ckks_vector(
        public_context,
        array.tolist(),
    )

    return encrypted.serialize()


def aggregate_encrypted_updates(
    encrypted_updates: list[bytes],
) -> np.ndarray:
    """Homomorphically add ciphertexts and decrypt only the aggregate."""

    if not encrypted_updates:
        raise ValueError(
            "At least one encrypted update is required"
        )

    private_context = load_private_context()

    aggregate = ts.ckks_vector_from(
        private_context,
        encrypted_updates[0],
    )

    for payload in encrypted_updates[1:]:
        current = ts.ckks_vector_from(
            private_context,
            payload,
        )
        aggregate += current

    decrypted = aggregate.decrypt()

    return np.asarray(
        decrypted,
        dtype=np.float32,
    )