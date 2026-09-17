from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import tenseal as ts


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

PUBLIC_CONTEXT_FILE = (
    DATA_DIR / "tenseal_public_context.bin"
)

SECRET_CONTEXT_FILE = (
    DATA_DIR / "tenseal_secret_context.bin"
)

POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
GLOBAL_SCALE = 2**40

# Keep chunks comfortably below CKKS slot capacity.
CKKS_CHUNK_SIZE = 2048


def create_context() -> ts.Context:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )

    context.global_scale = GLOBAL_SCALE

    return context


def ensure_context_files() -> None:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        SECRET_CONTEXT_FILE.exists()
        and PUBLIC_CONTEXT_FILE.exists()
    ):
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


def encrypt_update(
    values: np.ndarray | Iterable[float],
) -> bytes:
    public_context = load_public_context()

    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if len(array) > CKKS_CHUNK_SIZE:
        raise ValueError(
            "Update exceeds CKKS chunk size. "
            "Use encrypt_update_chunks()."
        )

    encrypted = ts.ckks_vector(
        public_context,
        array.tolist(),
    )

    return encrypted.serialize()


def encrypt_update_chunks(
    values: np.ndarray | Iterable[float],
) -> list[bytes]:
    array = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(-1)

    encrypted_chunks: list[bytes] = []

    for start in range(
        0,
        len(array),
        CKKS_CHUNK_SIZE,
    ):
        chunk = array[
            start:start + CKKS_CHUNK_SIZE
        ]

        encrypted_chunks.append(
            encrypt_update(chunk)
        )

    return encrypted_chunks


def aggregate_encrypted_updates(
    encrypted_updates: list[bytes],
) -> np.ndarray:
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

    return np.asarray(
        aggregate.decrypt(),
        dtype=np.float32,
    )


def aggregate_encrypted_chunks(
    client_updates: list[list[bytes]],
) -> np.ndarray:
    if not client_updates:
        raise ValueError(
            "At least one client update is required"
        )

    chunk_count = len(
        client_updates[0]
    )

    if chunk_count == 0:
        raise ValueError(
            "Client update contains no chunks"
        )

    for update in client_updates:
        if len(update) != chunk_count:
            raise ValueError(
                "All clients must have the same "
                "number of encrypted chunks"
            )

    aggregated_chunks: list[np.ndarray] = []

    for chunk_index in range(
        chunk_count
    ):
        chunk_payloads = [
            client_chunks[chunk_index]
            for client_chunks in client_updates
        ]

        aggregated_chunks.append(
            aggregate_encrypted_updates(
                chunk_payloads
            )
        )

    return np.concatenate(
        aggregated_chunks
    )