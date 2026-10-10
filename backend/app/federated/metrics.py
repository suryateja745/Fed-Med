from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
)

STATE_FILE = (
    DATA_DIR / "training_status.json"
)

DEFAULT_TOTAL_ROUNDS = 3

DEFAULT_HOSPITALS = {
    "hospital-1": "disconnected",
    "hospital-2": "disconnected",
    "hospital-3": "disconnected",
}


def _default_security() -> dict[str, Any]:
    return {
        "encryption": "TenSEAL CKKS",
        "encryption_enabled": True,
        "secure_aggregation": True,
        "encrypted_updates": 0,
        "plaintext_updates_exposed": False,
    }


def _default_state() -> dict[str, Any]:
    """Return the initial federated-learning runtime state."""
    return {
        "status": "idle",
        "current_round": 0,
        "total_rounds": DEFAULT_TOTAL_ROUNDS,
        "hospitals": DEFAULT_HOSPITALS.copy(),
        "retries": {
            "hospital-1": 0,
            "hospital-2": 0,
            "hospital-3": 0,
        },
        "failures": [],
        "rounds": [],
        "hospital_metrics": {},
        "security": _default_security(),
    }


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def _read_state() -> dict[str, Any]:
    """Read persisted state and add missing fields for compatibility."""
    _ensure_data_dir()

    if not STATE_FILE.exists():
        state = _default_state()
        _write_state(state)
        return state

    try:
        with STATE_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            state = json.load(file)

        if not isinstance(state, dict):
            state = _default_state()

    except (
        json.JSONDecodeError,
        OSError,
    ):
        state = _default_state()

    state.setdefault("status", "idle")
    state.setdefault("current_round", 0)
    state.setdefault(
        "total_rounds",
        DEFAULT_TOTAL_ROUNDS,
    )
    state.setdefault(
        "hospitals",
        DEFAULT_HOSPITALS.copy(),
    )
    state.setdefault(
        "retries",
        {
            "hospital-1": 0,
            "hospital-2": 0,
            "hospital-3": 0,
        },
    )
    state.setdefault("failures", [])
    state.setdefault("rounds", [])
    state.setdefault("hospital_metrics", {})
    state.setdefault(
        "security",
        _default_security(),
    )

    return state


def _write_state(
    state: dict[str, Any],
) -> None:
    """Write state through a temporary file to avoid partial JSON writes."""
    _ensure_data_dir()

    temporary_file = STATE_FILE.with_suffix(".tmp")

    with temporary_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            state,
            file,
            indent=2,
        )

    temporary_file.replace(STATE_FILE)


def reset_training_state() -> None:
    """Reset federated-learning runtime state."""
    _write_state(_default_state())


def get_training_status() -> dict[str, Any]:
    """Return the current persisted federated-learning state."""
    return _read_state()


get_training_state = get_training_status


def set_training_status(
    status: str,
    current_round: int | None = None,
) -> None:
    """Update the overall training status and optionally the round."""
    state = _read_state()

    state["status"] = status

    if current_round is not None:
        state["current_round"] = int(current_round)

    _write_state(state)


def set_hospital_status(
    hospital_id: str,
    status: str,
    round_number: int | None = None,
) -> None:
    """Update a hospital's status and record timeout events."""
    state = _read_state()

    hospitals = state.setdefault(
        "hospitals",
        {},
    )

    hospitals[hospital_id] = status

    if round_number is not None:
        state["current_round"] = max(
            int(state.get("current_round", 0)),
            int(round_number),
        )

    if status == "timeout":
        retries = state.setdefault(
            "retries",
            {},
        )

        retries[hospital_id] = (
            int(retries.get(hospital_id, 0)) + 1
        )

        failures = state.setdefault(
            "failures",
            [],
        )

        failures.append(
            {
                "hospital_id": hospital_id,
                "round": round_number,
                "status": "timeout",
                "retry_count": retries[hospital_id],
            }
        )

    _write_state(state)


def record_hospital_metrics(
    hospital_id: str,
    round_number: int,
    metrics: dict[str, Any],
) -> None:
    """
    Persist measured training metrics for one hospital and round.

    Missing numeric metrics remain absent; they are never replaced
    with invented loss or Dice values.
    """
    state = _read_state()

    hospital_metrics = state.setdefault(
        "hospital_metrics",
        {},
    )

    history = hospital_metrics.setdefault(
        hospital_id,
        [],
    )

    retries = state.get(
        "retries",
        {},
    ).get(hospital_id, 0)

    record: dict[str, Any] = {
        "round": int(round_number),
        "status": str(
            metrics.get("status", "trained")
        ),
        "retry_count": int(
            metrics.get("retry_count", retries)
        ),
    }

    numeric_fields = (
        "train_loss",
        "val_loss",
        "dice",
        "dp_max_norm",
        "dp_noise_multiplier",
    )

    for key in numeric_fields:
        value = metrics.get(key)

        if isinstance(value, (int, float)) and not isinstance(
            value,
            bool,
        ):
            record[key] = float(value)

    # Update the record for this round if one already exists.
    for index, previous in enumerate(history):
        if int(previous.get("round", -1)) == record["round"]:
            history[index] = record
            break
    else:
        history.append(record)

    history.sort(
        key=lambda item: int(item.get("round", 0))
    )

    state["hospital_metrics"] = hospital_metrics

    _write_state(state)


def record_security_updates(
    encrypted_updates: int,
) -> None:
    """Record encrypted-update counts reported by the training strategy."""
    state = _read_state()

    security = state.setdefault(
        "security",
        _default_security(),
    )

    security["encryption"] = "TenSEAL CKKS"
    security["encryption_enabled"] = True
    security["secure_aggregation"] = True

    security["encrypted_updates"] = (
        int(security.get("encrypted_updates", 0))
        + int(encrypted_updates)
    )

    security["plaintext_updates_exposed"] = False

    _write_state(state)


def record_round(
    round_number: int,
    loss: float,
    status: str = "completed",
) -> None:
    """Record or update the global loss and status for a round."""
    state = _read_state()

    rounds = state.setdefault(
        "rounds",
        [],
    )

    round_number = int(round_number)

    round_record = {
        "round": round_number,
        "status": status,
        "loss": float(loss),
    }

    existing_index = None

    for index, item in enumerate(rounds):
        if int(item.get("round", -1)) == round_number:
            existing_index = index
            break

    if existing_index is None:
        rounds.append(round_record)
    else:
        rounds[existing_index] = round_record

    rounds.sort(
        key=lambda item: int(item.get("round", 0))
    )

    state["current_round"] = max(
        int(state.get("current_round", 0)),
        round_number,
    )

    _write_state(state)