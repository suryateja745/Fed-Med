import requests
import pandas as pd
import streamlit as st


API_URL = "http://127.0.0.1:8000/api/training/status"


st.set_page_config(
    page_title="FedMed - Hospital Nodes",
    page_icon="H",
    layout="wide",
)


# -------------------------------------------------------------------
# Backend API
# -------------------------------------------------------------------

def get_training_status():
    """Fetch current federated training state from FastAPI."""
    try:
        response = requests.get(API_URL, timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


training_data = get_training_status()


# -------------------------------------------------------------------
# Hospital definitions
# -------------------------------------------------------------------

default_hospitals = [
    {
        "id": "hospital-1",
        "name": "Hospital 1",
        "dataset": "mock-dataset-1",
    },
    {
        "id": "hospital-2",
        "name": "Hospital 2",
        "dataset": "mock-dataset-2",
    },
    {
        "id": "hospital-3",
        "name": "Hospital 3",
        "dataset": "mock-dataset-3",
    },
]


def normalize_hospital_status(status):
    """Convert backend status into a consistent UI status."""

    value = str(status or "").strip().lower()

    mapping = {
        "connected": "Connected",
        "ready": "Connected",
        "online": "Connected",
        "training": "Training",
        "running": "Training",
        "trained": "Trained",
        "completed": "Trained",
        "retry": "Retrying",
        "retrying": "Retrying",
        "timeout": "Timeout",
        "failed": "Failed",
        "failure": "Failed",
        "disconnected": "Disconnected",
        "offline": "Disconnected",
    }

    return mapping.get(value, "Disconnected")


# -------------------------------------------------------------------
# Training state
# -------------------------------------------------------------------

backend_hospitals = {}
retries = {}
failures = []
rounds_data = []

current_round = 0
total_rounds = 3
training_status = "idle"


if training_data:
    backend_hospitals = (
        training_data.get("hospitals", {}) or {}
    )

    retries = (
        training_data.get("retries", {}) or {}
    )

    failures = (
        training_data.get("failures", []) or []
    )

    rounds_data = (
        training_data.get("rounds", []) or []
    )

    training_status = str(
        training_data.get("status", "idle")
    ).lower()

    try:
        current_round = int(
            training_data.get("current_round", 0)
        )
    except (TypeError, ValueError):
        current_round = 0

    try:
        total_rounds = int(
            training_data.get("total_rounds", 3)
        )
    except (TypeError, ValueError):
        total_rounds = 3


# -------------------------------------------------------------------
# Normalize round history
# -------------------------------------------------------------------

def normalize_rounds(data):
    """Support both list and dictionary round formats."""

    normalized = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                normalized.append(item)

    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                item = dict(value)

                if "round" not in item:
                    try:
                        item["round"] = int(key)
                    except (TypeError, ValueError):
                        pass

                normalized.append(item)

    normalized.sort(
        key=lambda item: int(item.get("round", 0))
        if str(item.get("round", "")).isdigit()
        else 0
    )

    return normalized


round_history = normalize_rounds(rounds_data)


def latest_loss(rounds):
    """Return the most recent available round loss."""

    for item in reversed(rounds):
        loss = item.get("loss")

        if loss is not None:
            try:
                return float(loss)
            except (TypeError, ValueError):
                return None

    return None


global_loss = latest_loss(round_history)

completed_rounds = sum(
    str(item.get("status", "")).lower()
    in {"completed", "complete", "success"}
    for item in round_history
)


# -------------------------------------------------------------------
# Hospital records
# -------------------------------------------------------------------

hospitals = []

for hospital in default_hospitals:
    hospital_id = hospital["id"]

    raw_status = backend_hospitals.get(
        hospital_id,
        "disconnected",
    )

    hospital_status = normalize_hospital_status(raw_status)

    try:
        retry_count = int(
            retries.get(hospital_id, 0)
        )
    except (TypeError, ValueError):
        retry_count = 0

    hospitals.append(
        {
            **hospital,
            "status": hospital_status,
            "retry_count": retry_count,
        }
    )


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.title("FedMed Hospital Node Management")

st.caption(
    "Cross-Silo Federated Learning - Mock Hospital Network"
)


refresh_col, status_col = st.columns([1, 5])

with refresh_col:
    if st.button(
        "Refresh Training Data",
        use_container_width=True,
    ):
        st.rerun()


with status_col:
    if training_data is None:
        st.error(
            "Backend API unavailable. "
            "Start the FastAPI server on port 8000."
        )
    else:
        st.info(
            f"Backend training status: "
            f"{training_status.upper()}"
        )


# -------------------------------------------------------------------
# Hospital Nodes
# -------------------------------------------------------------------

st.subheader("Federated Hospital Nodes")

cols = st.columns(3)


for col, hospital in zip(cols, hospitals):

    with col:

        st.markdown(
            f"### {hospital['name']}"
        )

        st.write(
            f"Node ID: `{hospital['id']}`"
        )

        st.write(
            f"Dataset: `{hospital['dataset']}`"
        )

        status = hospital["status"]

        if status == "Trained":
            st.success("TRAINED")

        elif status == "Training":
            st.info("TRAINING")

        elif status == "Retrying":
            st.warning("RETRYING")

        elif status == "Timeout":
            st.error("TIMEOUT")

        elif status == "Failed":
            st.error("FAILED")

        elif status == "Connected":
            st.success("CONNECTED")

        else:
            st.error("DISCONNECTED")

        st.metric(
            "Retries",
            hospital["retry_count"],
        )

        st.divider()

        if st.button(
            "Start Training",
            key=f"train_{hospital['id']}",
            use_container_width=True,
        ):
            st.info(
                f"{hospital['name']} "
                "training requested."
            )

        if st.button(
            "Reconnect",
            key=f"reconnect_{hospital['id']}",
            use_container_width=True,
        ):
            st.info(
                f"{hospital['name']} "
                "reconnect requested."
            )

        if st.button(
            "Disconnect",
            key=f"disconnect_{hospital['id']}",
            use_container_width=True,
        ):
            st.warning(
                f"{hospital['name']} "
                "disconnect requested."
            )


# -------------------------------------------------------------------
# Network Summary
# -------------------------------------------------------------------

st.divider()

st.subheader("Network Summary")

problem_statuses = {
    "Timeout",
    "Failed",
    "Disconnected",
}

active_count = sum(
    hospital["status"] not in problem_statuses
    for hospital in hospitals
)

problem_count = sum(
    hospital["status"] in problem_statuses
    for hospital in hospitals
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Total Hospitals",
        len(hospitals),
    )

with c2:
    st.metric(
        "Active / Participating Nodes",
        active_count,
    )

with c3:
    st.metric(
        "Problem Nodes",
        problem_count,
    )


# -------------------------------------------------------------------
# Federated Training Overview
# -------------------------------------------------------------------

st.divider()

st.subheader("Federated Training")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Current Round",
        f"{current_round}/{total_rounds}",
    )

with c2:
    st.metric(
        "Completed Rounds",
        completed_rounds,
    )

with c3:
    st.metric(
        "Global Loss",
        f"{global_loss:.2f}"
        if global_loss is not None
        else "—",
    )

with c4:
    st.metric(
        "Aggregation",
        "FedAvg",
    )


# -------------------------------------------------------------------
# Training Progress
# -------------------------------------------------------------------

st.markdown("### Training Progress")

try:
    total_for_progress = max(int(total_rounds), 1)

    progress_value = min(
        max(
            current_round / total_for_progress,
            0.0,
        ),
        1.0,
    )

    st.progress(
        progress_value
    )

except (TypeError, ValueError, ZeroDivisionError):
    pass


if training_status in {
    "completed",
    "complete",
    "success",
}:

    st.success(
        f"Federated training completed: "
        f"{current_round}/{total_rounds} rounds."
    )

elif training_status in {
    "running",
    "training",
}:

    st.info(
        f"Federated training is running "
        f"(Round {current_round}/{total_rounds})."
    )

else:

    st.info(
        "Federated training is ready."
    )


# -------------------------------------------------------------------
# Round History
# -------------------------------------------------------------------

st.divider()

st.subheader("Round History")

if round_history:

    table_rows = []

    for item in round_history:

        round_number = item.get(
            "round",
            "—",
        )

        round_status = item.get(
            "status",
            "Unknown",
        )

        loss = item.get(
            "loss"
        )

        if loss is not None:
            try:
                loss_display = round(
                    float(loss),
                    4,
                )
            except (TypeError, ValueError):
                loss_display = "—"
        else:
            loss_display = "—"

        table_rows.append(
            {
                "Round": round_number,
                "Status": str(
                    round_status
                ).title(),
                "Loss": loss_display,
            }
        )

    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No completed round history is available yet."
    )


# -------------------------------------------------------------------
# Loss Chart
# -------------------------------------------------------------------

st.markdown("### Global Loss by Round")

chart_rows = []

for item in round_history:

    round_number = item.get(
        "round"
    )

    loss = item.get(
        "loss"
    )

    try:
        round_number = int(
            round_number
        )
        loss = float(loss)

        chart_rows.append(
            {
                "Round": round_number,
                "Loss": loss,
            }
        )

    except (TypeError, ValueError):
        continue


if chart_rows:

    chart_df = (
        pd.DataFrame(chart_rows)
        .sort_values("Round")
        .set_index("Round")
    )

    st.line_chart(
        chart_df[
            ["Loss"]
        ],
        use_container_width=True,
    )

else:

    st.info(
        "Loss chart will appear after training rounds are recorded."
    )


# -------------------------------------------------------------------
# Hospital Local Metrics
# -------------------------------------------------------------------

st.divider()

st.subheader("Hospital Local Metrics")

local_rows = []

for hospital in hospitals:

    local_rows.append(
        {
            "Hospital": hospital["name"],
            "Node ID": hospital["id"],
            "Status": hospital["status"],
            "Retries": hospital["retry_count"],
        }
    )


st.dataframe(
    local_rows,
    use_container_width=True,
    hide_index=True,
)


# -------------------------------------------------------------------
# Node Failure / Retry Events
# -------------------------------------------------------------------

st.divider()

st.subheader("Node Failure & Retry Events")

if failures:

    failure_rows = []

    for failure in failures:

        if not isinstance(
            failure,
            dict,
        ):
            continue

        failure_rows.append(
            {
                "Hospital": failure.get(
                    "hospital_id",
                    "Unknown",
                ),
                "Round": failure.get(
                    "round",
                    "—",
                ),
                "Status": str(
                    failure.get(
                        "status",
                        "failure",
                    )
                ).upper(),
                "Retry Count": failure.get(
                    "retry_count",
                    0,
                ),
            }
        )

    if failure_rows:

        st.dataframe(
            failure_rows,
            use_container_width=True,
            hide_index=True,
        )

        for failure in failure_rows:

            st.warning(
                f"{failure['Hospital']} - "
                f"{failure['Status']} in "
                f"Round {failure['Round']} "
                f"(Retries: "
                f"{failure['Retry Count']})"
            )

    else:

        st.success(
            "No node failures recorded."
        )

else:

    st.success(
        "No node failures recorded."
    )