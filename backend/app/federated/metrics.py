from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
STATE_FILE = DATA_DIR / "training_status.json"

_LOCK = Lock()


def _default_state() -> dict:
    return {
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


def _read_state() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not STATE_FILE.exists():
        state = _default_state()
        _write_state(state)
        return state

    try:
        with STATE_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        state = _default_state()
        _write_state(state)
        return state


def _write_state(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    temp_file = STATE_FILE.with_suffix(".tmp")

    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)

    temp_file.replace(STATE_FILE)


def reset_training_state() -> None:
    with _LOCK:
        _write_state(_default_state())


def get_training_status() -> dict:
    with _LOCK:
        return _read_state()


def set_training_status(status: str) -> None:
    with _LOCK:
        state = _read_state()
        state["status"] = status
        _write_state(state)


def set_hospital_status(hospital_id: str, status: str) -> None:
    with _LOCK:
        state = _read_state()
        state.setdefault("hospitals", {})[hospital_id] = status
        _write_state(state)


def record_round(round_number: int, loss: float | None) -> None:
    with _LOCK:
        state = _read_state()
        state["current_round"] = round_number

        rounds = state.setdefault("rounds", [])

        round_entry = {
            "round": round_number,
            "status": "completed",
            "loss": loss,
        }

        existing = next(
            (
                item
                for item in rounds
                if item.get("round") == round_number
            ),
            None,
        )

        if existing is not None:
            existing.update(round_entry)
        else:
            rounds.append(round_entry)

        if round_number >= state.get("total_rounds", 3):
            state["status"] = "completed"

        _write_state(state)
get_training_state = get_training_status