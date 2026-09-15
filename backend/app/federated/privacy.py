from __future__ import annotations

import math

import numpy as np


def clip_update(
    values: np.ndarray,
    max_norm: float = 1.0,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    norm = float(np.linalg.norm(values))

    if norm <= max_norm or norm == 0.0:
        return values

    return values * (max_norm / norm)


def add_gaussian_noise(
    values: np.ndarray,
    noise_multiplier: float = 0.1,
    max_norm: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    noise_std = (
        noise_multiplier * max_norm
    )

    noise = rng.normal(
        0.0,
        noise_std,
        size=values.shape,
    ).astype(np.float32)

    return (
        np.asarray(values, dtype=np.float32)
        + noise
    )


def privatize_update(
    values: np.ndarray,
    max_norm: float = 1.0,
    noise_multiplier: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    clipped = clip_update(
        values,
        max_norm=max_norm,
    )

    return add_gaussian_noise(
        clipped,
        noise_multiplier=noise_multiplier,
        max_norm=max_norm,
        seed=seed,
    )