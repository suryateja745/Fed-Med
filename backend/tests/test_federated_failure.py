from app.federated.metrics import (
    get_training_status,
    record_round,
    reset_training_state,
    set_hospital_status,
)


def setup_function():
    reset_training_state()


def teardown_function():
    reset_training_state()


def test_hospital_timeout_is_recorded():
    set_hospital_status(
        "hospital-3",
        "timeout",
        round_number=2,
    )

    state = get_training_status()

    assert (
        state["hospitals"]["hospital-3"]
        == "timeout"
    )

    assert (
        state["retries"]["hospital-3"]
        == 1
    )

    assert len(state["failures"]) == 1


def test_multiple_timeouts_increment_retry_count():
    set_hospital_status(
        "hospital-3",
        "timeout",
        round_number=1,
    )

    set_hospital_status(
        "hospital-3",
        "timeout",
        round_number=2,
    )

    state = get_training_status()

    assert (
        state["retries"]["hospital-3"]
        == 2
    )

    assert len(state["failures"]) == 2


def test_round_can_complete_with_failure():
    set_hospital_status(
        "hospital-3",
        "timeout",
        round_number=2,
    )

    record_round(
        round_number=2,
        loss=0.5,
        status="completed_with_failure",
    )

    state = get_training_status()

    round_two = next(
        item
        for item in state["rounds"]
        if item["round"] == 2
    )

    assert (
        round_two["status"]
        == "completed_with_failure"
    )

    assert round_two["loss"] == 0.5