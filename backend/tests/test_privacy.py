import numpy as np

from app.federated.privacy import (
    add_gaussian_noise,
    clip_update,
    privatize_update,
)


def test_clip_update_respects_norm() -> None:
    values = np.array(
        [3.0, 4.0],
        dtype=np.float32,
    )

    result = clip_update(
        values,
        max_norm=1.0,
    )

    assert np.linalg.norm(result) <= 1.000001


def test_noise_changes_update() -> None:
    values = np.zeros(
        10,
        dtype=np.float32,
    )

    result = add_gaussian_noise(
        values,
        noise_multiplier=0.5,
        seed=42,
    )

    assert not np.allclose(
        result,
        values,
    )


def test_privatize_update() -> None:
    values = np.ones(
        8,
        dtype=np.float32,
    )

    result = privatize_update(
        values,
        max_norm=1.0,
        noise_multiplier=0.1,
        seed=42,
    )

    assert result.shape == values.shape