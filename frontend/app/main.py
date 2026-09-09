import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/training/status"

st.set_page_config(
    page_title="FedMed - Hospital Nodes",
    page_icon="H",
    layout="wide",
)

st.title("FedMed Hospital Node Management")
st.caption("Cross-Silo Federated Learning - Mock Hospital Network")


def get_training_status():
    """Fetch live federated training status from the backend."""
    try:
        response = requests.get(API_URL, timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


training_data = get_training_status()


# -------------------------------------------------------------------
# Hospital data
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
    """Convert backend status values to UI-friendly status."""
    if isinstance(status, str):
        value = status.strip().lower()
        if value in {"connected", "ready", "training", "completed"}:
            return "Connected"
        return "Disconnected"

    if isinstance(status, bool):
        return "Connected" if status else "Disconnected"

    return "Disconnected"


backend_hospitals = {}

if training_data:
    backend_hospitals = training_data.get("hospitals", {}) or {}


hospitals = []

for hospital in default_hospitals:
    hospital_id = hospital["id"]

    backend_status = backend_hospitals.get(
        hospital_id,
        "connected",
    )

    hospitals.append(
        {
            **hospital,
            "status": normalize_hospital_status(backend_status),
        }
    )


# -------------------------------------------------------------------
# Header controls
# -------------------------------------------------------------------

refresh_col, status_col = st.columns([1, 5])

with refresh_col:
    if st.button("Refresh Status", use_container_width=True):
        st.rerun()

with status_col:
    if training_data is None:
        st.warning(
            "Backend API is unavailable. Start the FastAPI server on port 8000."
        )
    else:
        backend_status = training_data.get("status", "unknown")
        st.info(f"Backend training status: {str(backend_status).upper()}")


# -------------------------------------------------------------------
# Hospital Nodes
# -------------------------------------------------------------------

st.subheader("Federated Hospital Nodes")

cols = st.columns(3)

for col, hospital in zip(cols, hospitals):
    with col:
        st.markdown(f"### {hospital['name']}")
        st.write(f"Node ID: `{hospital['id']}`")
        st.write(f"Dataset: `{hospital['dataset']}`")

        if hospital["status"] == "Connected":
            st.success("CONNECTED")
        else:
            st.error("DISCONNECTED")

        st.divider()

        if st.button(
            "Start Training",
            key=f"train_{hospital['id']}",
            use_container_width=True,
        ):
            st.info(f"{hospital['name']} training started.")

        if st.button(
            "Reconnect",
            key=f"reconnect_{hospital['id']}",
            use_container_width=True,
        ):
            st.info(f"{hospital['name']} reconnect requested.")

        if st.button(
            "Disconnect",
            key=f"disconnect_{hospital['id']}",
            use_container_width=True,
        ):
            st.warning(f"{hospital['name']} disconnect requested.")


# -------------------------------------------------------------------
# Network Summary
# -------------------------------------------------------------------

st.divider()
st.subheader("Network Summary")

connected_count = sum(
    hospital["status"] == "Connected"
    for hospital in hospitals
)

disconnected_count = len(hospitals) - connected_count

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Total Hospitals", len(hospitals))

with c2:
    st.metric("Connected Nodes", connected_count)

with c3:
    st.metric("Disconnected Nodes", disconnected_count)


# -------------------------------------------------------------------
# Federated Training
# -------------------------------------------------------------------

st.divider()
st.subheader("Federated Training")

current_round = 0
total_rounds = 3
training_status = "idle"
hospitals_participating = connected_count
rounds_data = {}

if training_data:
    training_status = training_data.get("status", "idle")
    current_round = training_data.get("current_round", 0)
    total_rounds = training_data.get("total_rounds", 3)
    rounds_data = training_data.get("rounds", {}) or {}

    if training_status in {"running", "training"}:
        hospitals_participating = connected_count


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Training Rounds", total_rounds)

with col2:
    st.metric("Hospitals Participating", hospitals_participating)

with col3:
    st.metric("Aggregation", "FedAvg")

st.markdown("### Training Progress")


def get_round_info(rounds_data, round_number):
    """Return the requested round from either a list or dictionary."""

    if isinstance(rounds_data, list):
        for item in rounds_data:
            if isinstance(item, dict):
                try:
                    item_round = int(item.get("round", -1))
                except (TypeError, ValueError):
                    item_round = -1

                if item_round == round_number:
                    return item

        return None

    if isinstance(rounds_data, dict):
        return rounds_data.get(str(round_number)) or rounds_data.get(
            round_number
        )

    return None


for round_number in range(1, int(total_rounds) + 1):
    round_info = get_round_info(rounds_data, round_number)

    if isinstance(round_info, dict):
        round_status = round_info.get("status", "Pending")
        loss = round_info.get("loss")

        if loss is not None:
            st.write(
                f"**Round {round_number}** — "
                f"{round_status} | Loss: {loss}"
            )
        else:
            st.write(
                f"**Round {round_number}** — {round_status}"
            )

    elif isinstance(round_info, str):
        st.write(
            f"**Round {round_number}** — {round_info}"
        )

    else:
        if round_number < int(current_round):
            status = "Completed"
        elif (
            round_number == int(current_round)
            and int(current_round) > 0
        ):
            status = "Running"
        else:
            status = "Pending"

        st.write(
            f"**Round {round_number}** — {status}"
        )


if str(training_status).lower() in {"completed", "complete", "success"}:
    st.success(
        "Federated training completed successfully across all 3 hospitals."
    )
elif str(training_status).lower() in {"running", "training"}:
    st.info(
        f"Federated training is currently running "
        f"(Round {current_round}/{total_rounds})."
    )
else:
    st.info(
        "Federated training is ready. Start a training round to begin."
    )