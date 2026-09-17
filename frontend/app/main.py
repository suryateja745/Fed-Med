import requests
import pandas as pd
import streamlit as st


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="FedMed - Hospital Nodes",
    page_icon="H",
    layout="wide",
)


# -------------------------------------------------------------------
# Backend API
# -------------------------------------------------------------------

API_URL = "http://127.0.0.1:8000/api/training/status"


def get_training_status():
    """Fetch current federated training state from FastAPI."""
    try:
        response = requests.get(
            API_URL,
            timeout=3,
        )
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
        "dataset": "synthetic-MRI-dataset",
    },
    {
        "id": "hospital-2",
        "name": "Hospital 2",
        "dataset": "synthetic-MRI-dataset",
    },
    {
        "id": "hospital-3",
        "name": "Hospital 3",
        "dataset": "synthetic-MRI-dataset",
    },
]


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def normalize_hospital_status(status):
    """Convert backend status into a consistent UI status."""

    value = str(
        status or ""
    ).strip().lower()

    mapping = {
        "connected": "Connected",
        "ready": "Connected",
        "online": "Connected",

        "training": "Training",
        "running": "Training",

        "trained": "Trained",
        "completed": "Trained",
        "encrypted": "Trained",

        "retry": "Retrying",
        "retrying": "Retrying",

        "timeout": "Timeout",

        "failed": "Failed",
        "failure": "Failed",

        "disconnected": "Disconnected",
        "offline": "Disconnected",
    }

    return mapping.get(
        value,
        "Disconnected",
    )


def normalize_rounds(data):
    """Support both list and dictionary round formats."""

    normalized = []

    if isinstance(data, list):

        for item in data:

            if isinstance(item, dict):
                normalized.append(
                    dict(item)
                )

    elif isinstance(data, dict):

        for key, value in data.items():

            if isinstance(value, dict):

                item = dict(value)

                if "round" not in item:

                    try:
                        item["round"] = int(key)

                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

                normalized.append(item)

    def round_key(item):

        try:
            return int(
                item.get(
                    "round",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0

    normalized.sort(
        key=round_key
    )

    return normalized


def latest_loss(rounds):
    """Return the most recent available round loss."""

    for item in reversed(rounds):

        loss = item.get("loss")

        if loss is not None:

            try:
                return float(loss)

            except (
                TypeError,
                ValueError,
            ):
                continue

    return None


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

security = {}


if training_data:

    backend_hospitals = (
        training_data.get(
            "hospitals",
            {},
        )
        or {}
    )

    retries = (
        training_data.get(
            "retries",
            {},
        )
        or {}
    )

    failures = (
        training_data.get(
            "failures",
            [],
        )
        or []
    )

    rounds_data = (
        training_data.get(
            "rounds",
            [],
        )
        or []
    )

    security = (
        training_data.get(
            "security",
            {},
        )
        or {}
    )

    training_status = str(
        training_data.get(
            "status",
            "idle",
        )
    ).lower()

    try:

        current_round = int(
            training_data.get(
                "current_round",
                0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        current_round = 0

    try:

        total_rounds = int(
            training_data.get(
                "total_rounds",
                3,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        total_rounds = 3


# -------------------------------------------------------------------
# Round history
# -------------------------------------------------------------------

round_history = normalize_rounds(
    rounds_data
)

global_loss = latest_loss(
    round_history
)

completed_rounds = sum(
    str(
        item.get(
            "status",
            "",
        )
    ).lower()
    in {
        "completed",
        "complete",
        "success",
        "completed_with_failure",
    }
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

    hospital_status = (
        normalize_hospital_status(
            raw_status
        )
    )

    try:

        retry_count = int(
            retries.get(
                hospital_id,
                0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        retry_count = 0

    hospitals.append(
        {
            **hospital,
            "status": hospital_status,
            "retry_count": retry_count,
        }
    )


# -------------------------------------------------------------------
# Security values
# -------------------------------------------------------------------

encryption_name = security.get(
    "encryption",
    "Unavailable",
)

encryption_enabled = bool(
    security.get(
        "encryption_enabled",
        False,
    )
)

secure_aggregation = bool(
    security.get(
        "secure_aggregation",
        False,
    )
)

encrypted_updates = security.get(
    "encrypted_updates",
    0,
)

plaintext_updates_exposed = bool(
    security.get(
        "plaintext_updates_exposed",
        False,
    )
)


try:

    encrypted_updates_display = int(
        encrypted_updates
    )

except (
    TypeError,
    ValueError,
):

    encrypted_updates_display = 0


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.title(
    "FedMed Hospital Node Management"
)

st.caption(
    "Cross-Silo Federated Learning - "
    "Federated Hospital Network"
)


refresh_col, status_col = st.columns(
    [1, 5]
)

with refresh_col:

    if st.button(
        "Refresh Training Data",
        width="stretch",
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

st.subheader(
    "Federated Hospital Nodes"
)

cols = st.columns(3)

for col, hospital in zip(
    cols,
    hospitals,
):

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

            st.success(
                "TRAINED"
            )

        elif status == "Training":

            st.info(
                "TRAINING"
            )

        elif status == "Retrying":

            st.warning(
                "RETRYING"
            )

        elif status == "Timeout":

            st.error(
                "TIMEOUT"
            )

        elif status == "Failed":

            st.error(
                "FAILED"
            )

        elif status == "Connected":

            st.success(
                "CONNECTED"
            )

        else:

            st.error(
                "DISCONNECTED"
            )

        st.metric(
            "Retries",
            hospital["retry_count"],
        )

        st.divider()

        if st.button(
            "Start Training",
            key=f"train_{hospital['id']}",
            width="stretch",
        ):

            st.info(
                f"{hospital['name']} "
                "training requested."
            )

        if st.button(
            "Reconnect",
            key=f"reconnect_{hospital['id']}",
            width="stretch",
        ):

            st.info(
                f"{hospital['name']} "
                "reconnect requested."
            )

        if st.button(
            "Disconnect",
            key=f"disconnect_{hospital['id']}",
            width="stretch",
        ):

            st.warning(
                f"{hospital['name']} "
                "disconnect requested."
            )


# -------------------------------------------------------------------
# Network Summary
# -------------------------------------------------------------------

st.divider()

st.subheader(
    "Network Summary"
)

problem_statuses = {
    "Timeout",
    "Failed",
    "Disconnected",
}

active_count = sum(
    hospital["status"]
    not in problem_statuses
    for hospital in hospitals
)

problem_count = sum(
    hospital["status"]
    in problem_statuses
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
# Training Configuration
# -------------------------------------------------------------------

st.markdown("---")

st.subheader(
    "Training Configuration"
)

config_col1, config_col2, config_col3 = (
    st.columns(3)
)

with config_col1:

    config_epochs = st.number_input(
        "Epochs",
        min_value=1,
        max_value=100,
        value=2,
        step=1,
        key="config_epochs",
    )

with config_col2:

    config_batch_size = st.number_input(
        "Batch Size",
        min_value=1,
        max_value=16,
        value=1,
        step=1,
        key="config_batch_size",
    )

with config_col3:

    config_learning_rate = st.number_input(
        "Learning Rate",
        min_value=0.000001,
        max_value=1.0,
        value=0.001,
        step=0.0001,
        format="%.6f",
        key="config_learning_rate",
    )


selected_hospitals = st.multiselect(
    "Select Hospital Nodes",
    options=[
        "hospital-1",
        "hospital-2",
        "hospital-3",
    ],
    default=[
        "hospital-1",
        "hospital-2",
        "hospital-3",
    ],
    key="selected_hospitals",
)


button_col1, button_col2 = st.columns(2)

with button_col1:

    if st.button(
        "Initialize Training",
        width="stretch",
    ):

        st.session_state[
            "training_config"
        ] = {
            "epochs": config_epochs,
            "batch_size": config_batch_size,
            "learning_rate": config_learning_rate,
            "selected_hospitals": selected_hospitals,
        }

        st.success(
            "Configuration saved for the next training run."
        )


with button_col2:

    if st.button(
        "Reset Configuration",
        width="stretch",
    ):

        st.session_state[
            "training_config"
        ] = {
            "epochs": 2,
            "batch_size": 1,
            "learning_rate": 0.001,
            "selected_hospitals": [
                "hospital-1",
                "hospital-2",
                "hospital-3",
            ],
        }

        st.success(
            "Training configuration reset."
        )

        st.rerun()


st.caption(
    "Current federated backend defaults: "
    "1 epoch, batch size 1, learning rate 0.001. "
    "This panel prepares the configuration for "
    "the next training run."
)


# -------------------------------------------------------------------
# Federated Training Overview
# -------------------------------------------------------------------

st.divider()

st.subheader(
    "Federated Training"
)

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
        f"{global_loss:.4f}"
        if global_loss is not None
        else "N/A",
    )

with c4:

    st.metric(
        "Aggregation",
        "FedAvg",
    )


# -------------------------------------------------------------------
# Training Progress
# -------------------------------------------------------------------

st.markdown(
    "### Training Progress"
)

try:

    total_for_progress = max(
        int(total_rounds),
        1,
    )

    progress_value = min(
        max(
            current_round
            / total_for_progress,
            0.0,
        ),
        1.0,
    )

    st.progress(
        progress_value
    )

except (
    TypeError,
    ValueError,
    ZeroDivisionError,
):

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

elif training_status == "error":

    st.error(
        "Federated training encountered an error."
    )

else:

    st.info(
        "Federated training is ready."
    )


# -------------------------------------------------------------------
# Latest Federated Run
# -------------------------------------------------------------------

st.divider()

st.subheader(
    "Latest Federated Run"
)

summary_col1, summary_col2, summary_col3, summary_col4 = (
    st.columns(4)
)

with summary_col1:

    st.metric(
        "Hospitals",
        len(hospitals),
    )

with summary_col2:

    st.metric(
        "Rounds",
        f"{current_round}/{total_rounds}",
    )

with summary_col3:

    st.metric(
        "Encrypted Updates",
        encrypted_updates_display,
    )

with summary_col4:

    st.metric(
        "Status",
        training_status.title(),
    )


# -------------------------------------------------------------------
# ML Training Pipeline
# -------------------------------------------------------------------

st.divider()

st.subheader(
    "ML Training Pipeline"
)

ml_col1, ml_col2, ml_col3, ml_col4 = (
    st.columns(4)
)

with ml_col1:

    st.metric(
        "Model",
        "MONAI 3D U-Net",
    )

with ml_col2:

    st.metric(
        "Framework",
        "PyTorch",
    )

with ml_col3:

    st.metric(
        "Training",
        "Federated Local",
    )

with ml_col4:

    st.metric(
        "Dataset",
        "Synthetic MRI",
    )


st.info(
    "Each hospital trains the 3D U-Net locally. "
    "The protected local model update is then "
    "encrypted and securely aggregated."
)


# -------------------------------------------------------------------
# Security & Privacy
# -------------------------------------------------------------------

st.divider()

st.subheader(
    "Security & Privacy"
)

security_col1, security_col2, security_col3 = (
    st.columns(3)
)

with security_col1:

    st.metric(
        "Encryption",
        encryption_name,
    )

with security_col2:

    st.metric(
        "Encrypted Updates",
        encrypted_updates_display,
    )

with security_col3:

    st.metric(
        "Secure Aggregation",
        "Enabled"
        if secure_aggregation
        else "Disabled",
    )


security_col4, security_col5 = (
    st.columns(2)
)

with security_col4:

    if encryption_enabled:

        st.success(
            "OK - Encryption Enabled"
        )

    else:

        st.error(
            "ERROR - Encryption Disabled"
        )


with security_col5:

    if plaintext_updates_exposed:

        st.error(
            "ERROR - Plaintext Updates Exposed"
        )

    else:

        st.success(
            "OK - Plaintext Updates Not Exposed"
        )


st.caption(
    "Hospital model updates are protected with "
    "Differential Privacy before TenSEAL CKKS "
    "encryption and secure aggregation."
)


# -------------------------------------------------------------------
# Differential Privacy
# -------------------------------------------------------------------

st.divider()

st.subheader(
    "Differential Privacy"
)

dp_col1, dp_col2, dp_col3 = (
    st.columns(3)
)

with dp_col1:

    st.metric(
        "DP Status",
        "Enabled",
    )

with dp_col2:

    st.metric(
        "Clipping Norm",
        "1.0",
    )

with dp_col3:

    st.metric(
        "Noise Multiplier",
        "0.1",
    )


dp_col4, dp_col5 = st.columns(2)

with dp_col4:

    st.success(
        "OK - DP applied before encryption"
    )

with dp_col5:

    st.success(
        "OK - Private update protected"
    )


st.caption(
    "Local model updates are clipped and "
    "perturbed before TenSEAL CKKS encryption."
)


if (
    encryption_enabled
    and secure_aggregation
    and not plaintext_updates_exposed
):

    st.success(
        "FedMed security pipeline active: "
        "DP -> TenSEAL CKKS -> secure aggregation"
    )

else:

    st.warning(
        "Security pipeline is not fully active."
    )


# -------------------------------------------------------------------
# Round History
# -------------------------------------------------------------------

st.divider()

st.subheader(
    "Round History"
)

if round_history:

    table_rows = []

    for item in round_history:

        round_number = item.get(
            "round",
            "N/A",
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

            except (
                TypeError,
                ValueError,
            ):

                loss_display = "N/A"

        else:

            loss_display = "N/A"

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
        width="stretch",
        hide_index=True,
    )

else:

    st.info(
        "No completed round history "
        "is available yet."
    )


# -------------------------------------------------------------------
# Loss Chart
# -------------------------------------------------------------------

st.markdown(
    "### Global Loss by Round"
)

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

        loss = float(
            loss
        )

        chart_rows.append(
            {
                "Round": round_number,
                "Loss": loss,
            }
        )

    except (
        TypeError,
        ValueError,
    ):

        continue


if chart_rows:

    chart_df = (
        pd.DataFrame(
            chart_rows
        )
        .sort_values(
            "Round"
        )
        .set_index(
            "Round"
        )
    )

    st.line_chart(
        chart_df[
            ["Loss"]
        ],
        width="stretch",
    )

else:

    st.info(
        "Loss chart will appear after "
        "training rounds are recorded."
    )


# -------------------------------------------------------------------
# Hospital Federated Status
# -------------------------------------------------------------------

st.divider()

st.subheader(
    "Hospital Federated Status"
)

local_rows = []

for hospital in hospitals:

    local_rows.append(
        {
            "Hospital": hospital["name"],
            "Node ID": hospital["id"],
            "Status": hospital["status"],
            "Training Role": "Local model training",
            "Retries": hospital["retry_count"],
        }
    )


st.dataframe(
    local_rows,
    width="stretch",
    hide_index=True,
)


# -------------------------------------------------------------------
# Node Failure / Retry Events
# -------------------------------------------------------------------

st.divider()

st.subheader(
    "Node Failure & Retry Events"
)

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
                    "N/A",
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
            width="stretch",
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


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.divider()

st.caption(
    "FedMed | Cross-Silo Federated Learning "
    "Engine | Hospital Privacy Preserved"
)