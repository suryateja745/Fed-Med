import numpy as np

from app.federated.secure_aggregation import (
    encrypt_update,
    aggregate_encrypted_updates,
)


def test_encrypted_update_is_bytes():
    values = np.array([0.1], dtype=np.float32)

    encrypted = encrypt_update(values)

    assert isinstance(encrypted, bytes)
    assert len(encrypted) > 0


def test_encrypted_updates_are_aggregated():
    update_1 = np.array([0.1], dtype=np.float32)
    update_2 = np.array([0.2], dtype=np.float32)
    update_3 = np.array([0.3], dtype=np.float32)

    encrypted_updates = [
        encrypt_update(update_1),
        encrypt_update(update_2),
        encrypt_update(update_3),
    ]

    result = aggregate_encrypted_updates(
        encrypted_updates
    )

    assert result.shape == (1,)
    assert np.isclose(
        float(result[0]),
        0.6,
        atol=0.01,
    )