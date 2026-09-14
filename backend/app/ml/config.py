from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    epochs: int = 2
    batch_size: int = 1
    learning_rate: float = 1e-3
    selected_hospitals: tuple[str, ...] = (
        "hospital-1",
        "hospital-2",
        "hospital-3",
    )