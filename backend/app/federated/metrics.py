from __future__ import annotations

from typing import Any


training_state: dict[str, Any] = {
    "status": "idle",
    "current_round": 0,
    "total_rounds": 3,
    "hospitals": {
        "hospital-1": "disconnected",
        "hospital-2": "disconnected",
        "hospital-3": "disconnected",
    },
    "rounds": [],
}


def reset_training() -> None:
    training_state["status"] = "idle"
    training_state["current_round"] = 0
    training_state["rounds"] = []

    for hospital_id in training_state["hospitals"]:
        training_state["hospitals"][hospital_id] = "disconnected"


def set_training_status(status: str) -> None:
    training_state["status"] = status


def set_hospital_status(hospital_id: str, status: str) -> None:
    if hospital_id in training_state["hospitals"]:
        training_state["hospitals"][hospital_id] = status


def record_round(
    round_number: int,
    loss: float | None = None,
    dice: float | None = None,
) -> None:
    training_state["current_round"] = round_number

    training_state["rounds"].append(
        {
            "round": round_number,
            "loss": loss,
            "dice": dice,
        }
    )


def get_training_state() -> dict[str, Any]:
    return training_state