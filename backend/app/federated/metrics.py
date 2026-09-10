from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_DIR = (
    Path(__file__).resolve().parents[2] / "data"
)

STATE_FILE = DATA_DIR / "training_status.json"

DEFAULT_TOTAL_ROUNDS = 3

DEFAULT_HOSPITALS = {
    "hospital-1": "disconnected",
    "hospital-2": "disconnected",
    "hospital-3": "disconnected",
}


def _default_state() -> dict[str, Any]:
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
    }


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def _read_state() -> dict[str, Any]:
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
    except (
        json.JSONDecodeError,
        OSError,
    ):
        state = _default_state()

    # Backward compatibility with the 08/09 JSON.
    state.setdefault(
        "status",
        "idle",
    )
    state.setdefault(
        "current_round",
        0,
    )
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
    state.setdefault(
        "failures",
        [],
    )
    state.setdefault(
        "rounds",
        [],
    )

    return state


def _write_state(
    state: dict[str, Any],
) -> None:
    _ensure_data_dir()

    temporary_file = STATE_FILE.with_suffix(
        ".tmp"
    )

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
    """Reset the persisted FL training state."""

    _write_state(_default_state())


def get_training_status() -> dict[str, Any]:
    """Return current persisted FL training state."""

    return _read_state()


# Compatibility alias used by older code.
get_training_state = get_training_status


def set_training_status(
    status: str,
    current_round: int | None = None,
) -> None:
    """Update global training status."""

    state = _read_state()

    state["status"] = status

    if current_round is not None:
        state["current_round"] = int(
            current_round
        )

    _write_state(state)


def set_hospital_status(
    hospital_id: str,
    status: str,
    round_number: int | None = None,
) -> None:
    """Update a hospital's current status."""

    state = _read_state()

    hospitals = state.setdefault(
        "hospitals",
        {},
    )

    # Preserve the old frontend-compatible format:
    #
    # "hospital-1": "trained"
    #
    hospitals[hospital_id] = status

    if round_number is not None:
        state["current_round"] = max(
            int(state.get("current_round", 0)),
            int(round_number),
        )

    # A successful training event does not clear historical
    # retry information.
    if status == "timeout":
        retries = state.setdefault(
            "retries",
            {},
        )

        retries[hospital_id] = (
            int(retries.get(hospital_id, 0))
            + 1
        )

        failures = state.setdefault(
            "failures",
            [],
        )

        failure_record = {
            "hospital_id": hospital_id,
            "round": round_number,
            "status": "timeout",
            "retry_count": retries[hospital_id],
        }

        failures.append(
            failure_record
        )

    _write_state(state)


def record_round(
    round_number: int,
    loss: float,
    status: str = "completed",
) -> None:
    """Persist metrics for one federated round."""

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

    # Update an existing round rather than duplicating it.
    existing_index = None

    for index, item in enumerate(rounds):
        if int(
            item.get("round", -1)
        ) == round_number:
            existing_index = index
            break

    if existing_index is None:
        rounds.append(round_record)
    else:
        rounds[existing_index] = round_record

    rounds.sort(
        key=lambda item: int(
            item.get("round", 0)
        )
    )

    state["current_round"] = max(
        int(state.get("current_round", 0)),
        round_number,
    )

    _write_state(state)