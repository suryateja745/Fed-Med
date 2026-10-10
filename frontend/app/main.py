
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st


# ---------------------------------------------------------
# FedMed dashboard configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="FedMed | Federated Learning Dashboard",
    page_icon="🏥",
    layout="wide",
)

API_BASE_URL = "http://127.0.0.1:8000"
HEALTH_URL = f"{API_BASE_URL}/health"
STATUS_URL = f"{API_BASE_URL}/api/training/status"

EXPECTED_HOSPITALS = [
    "hospital-1",
    "hospital-2",
    "hospital-3",
]


# ---------------------------------------------------------
# API helpers
# ---------------------------------------------------------

def fetch_json(url: str) -> tuple[dict | None, str | None]:
    """Fetch one JSON response from the local FastAPI server."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            return None, "The API response was not a JSON object."

        return data, None

    except requests.RequestException as exc:
        return None, str(exc)

    except ValueError:
        return None, "The API returned invalid JSON."


def safe_number(value, default=0):
    """Convert a numeric value safely for dashboard display."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def display_number(value, digits=4):
    """Display an available metric without inventing a value."""
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------
# Header and refresh
# ---------------------------------------------------------

st.title("🏥 FedMed — Federated Hospital Dashboard")

st.caption(
    "Cross-Silo Federated Learning Engine | "
    "Research prototype for privacy-aware medical image model training"
)

refresh_col, info_col = st.columns([1, 4])

with refresh_col:
    if st.button("Refresh dashboard", width="stretch"):
        st.rerun()

with info_col:
    st.caption(
        "Refresh after hospitals connect or a federated round finishes."
    )


# ---------------------------------------------------------
# Backend connection
# ---------------------------------------------------------

health_data, health_error = fetch_json(HEALTH_URL)
training_data, training_error = fetch_json(STATUS_URL)

if health_data and health_data.get("status") == "healthy":
    st.success("FastAPI backend: healthy")
else:
    st.error(
        "Backend unavailable. Start FastAPI on port 8000. "
        f"Details: {health_error or 'Unexpected health response.'}"
    )

if training_data is None:
    st.error(
        "Training state could not be loaded. "
        f"Details: {training_error}"
    )
    st.stop()


# ---------------------------------------------------------
# Read current runtime state
# ---------------------------------------------------------

training_status = str(
    training_data.get("status", "unknown")
).lower()

current_round = int(
    safe_number(training_data.get("current_round", 0))
)

total_rounds = int(
    safe_number(training_data.get("total_rounds", 3))
)

hospital_states = training_data.get("hospitals", {}) or {}
retry_states = training_data.get("retries", {}) or {}
rounds = training_data.get("rounds", []) or []
hospital_metrics = (
    training_data.get("hospital_metrics", {}) or {}
)
failures = training_data.get("failures", []) or []
security = training_data.get("security", {}) or {}

if not isinstance(rounds, list):
    rounds = []

if not isinstance(failures, list):
    failures = []

# Include configured hospitals even when they have not connected yet.
hospital_ids = list(
    dict.fromkeys(
        EXPECTED_HOSPITALS + list(hospital_states.keys())
    )
)

active_statuses = {
    "connected",
    "online",
    "ready",
    "training",
    "running",
    "trained",
    "completed",
}

active_count = sum(
    str(hospital_states.get(hospital_id, "disconnected")).lower()
    in active_statuses
    for hospital_id in hospital_ids
)


# ---------------------------------------------------------
# Main summary
# ---------------------------------------------------------

st.divider()
st.header("Federated Training Overview")

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric("Training Status", training_status.upper())

with metric2:
    st.metric(
        "Current Round",
        f"{current_round}/{total_rounds}",
    )

with metric3:
    st.metric("Hospitals", len(hospital_ids))

with metric4:
    st.metric("Active / Trained Nodes", active_count)

if total_rounds > 0:
    progress = min(max(current_round / total_rounds, 0.0), 1.0)
    st.progress(progress, text="Federated round progress")


# ---------------------------------------------------------
# Hospital status
# ---------------------------------------------------------

st.divider()
st.header("Hospital Nodes")

hospital_rows = []

for hospital_id in hospital_ids:
    hospital_rows.append({
        "Hospital ID": hospital_id,
        "Connection / Training Status": str(
            hospital_states.get(hospital_id, "disconnected")
        ),
        "Retry Count": retry_states.get(hospital_id, 0),
        "Metrics Recorded": len(
            hospital_metrics.get(hospital_id, []) or []
        ),
        "Dataset Mode": "Synthetic demo data",
    })

st.dataframe(
    pd.DataFrame(hospital_rows),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "A hospital may have no record for an individual round "
    "if it was not selected to participate in that round."
)


# ---------------------------------------------------------
# Global round history and loss
# ---------------------------------------------------------

st.divider()
st.header("Federated Round History")

if rounds:
    rounds_df = pd.DataFrame(rounds)

    display_columns = [
        column
        for column in ["round", "status", "loss"]
        if column in rounds_df.columns
    ]

    if display_columns:
        st.dataframe(
            rounds_df[display_columns],
            use_container_width=True,
            hide_index=True,
        )

    if {"round", "loss"}.issubset(rounds_df.columns):
        chart_df = rounds_df[["round", "loss"]].copy()
        chart_df["round"] = pd.to_numeric(
            chart_df["round"], errors="coerce"
        )
        chart_df["loss"] = pd.to_numeric(
            chart_df["loss"], errors="coerce"
        )
        chart_df = chart_df.dropna().sort_values("round")

        if not chart_df.empty:
            st.subheader("Global Loss by Round")
            st.line_chart(
                chart_df.set_index("round")["loss"]
            )
else:
    st.info(
        "No federated rounds have been recorded yet. "
        "Start the Flower server and hospital clients to run training."
    )


# ---------------------------------------------------------
# Per-hospital training metrics
# ---------------------------------------------------------

st.divider()
st.header("Per-Hospital Training Metrics")

metric_rows = []

if isinstance(hospital_metrics, dict):
    for hospital_id, history in hospital_metrics.items():
        if not isinstance(history, list):
            continue

        for item in history:
            if not isinstance(item, dict):
                continue

            metric_rows.append({
                "Hospital ID": hospital_id,
                "Round": item.get("round"),
                "Update Status": item.get("status", "unknown"),
                "Train Loss": item.get("train_loss"),
                "Validation Loss": item.get("val_loss"),
                "Dice Score": item.get("dice"),
                "Retries": item.get("retry_count", 0),
                "DP Max Norm": item.get("dp_max_norm"),
                "DP Noise Multiplier": item.get(
                    "dp_noise_multiplier"
                ),
            })

if metric_rows:
    metrics_df = pd.DataFrame(metric_rows)

    sort_columns = [
        column
        for column in ["Hospital ID", "Round"]
        if column in metrics_df.columns
    ]

    if sort_columns:
        metrics_df = metrics_df.sort_values(sort_columns)

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "No hospital-level training metrics are recorded yet. "
        "They will appear after clients participate in training."
    )

st.caption(
    "Dice scores and losses are measured outputs from the configured "
    "training pipeline; they are not a guarantee of clinical performance."
)


# ---------------------------------------------------------
# Failures and recovery information
# ---------------------------------------------------------

st.divider()
st.header("Failures and Recovery")

if failures:
    st.dataframe(
        pd.DataFrame(failures),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.success("No failures are recorded in the current runtime state.")


# ---------------------------------------------------------
# Security configuration reported by backend
# ---------------------------------------------------------

st.divider()
st.header("Security Configuration")


security_rows = [
    {
        "Setting": "Encryption method reported",
        "Value": str(security.get("encryption", "Unknown")),
    },
    {
        "Setting": "Encryption enabled flag",
        "Value": str(security.get("encryption_enabled", "Unknown")),
    },
    {
        "Setting": "Secure aggregation flag",
        "Value": str(security.get("secure_aggregation", "Unknown")),
    },
    {
        "Setting": "Encrypted updates recorded",
        "Value": str(security.get("encrypted_updates", 0)),
    },
    {
        "Setting": "Plaintext update exposure flag",
        "Value": str(security.get("plaintext_updates_exposed", "Unknown")),
    },
]

st.dataframe(
    pd.DataFrame(security_rows),
    use_container_width=True,
    hide_index=True,
)

st.warning(
    "These are backend-reported settings, not an independent security "
    "audit. Verify key ownership, decryption access, transport protection "
    "and privacy accounting before claiming server-blind secure aggregation "
    "or a formal differential-privacy guarantee."
)


# ---------------------------------------------------------
# Raw state for debugging and demonstration
# ---------------------------------------------------------

with st.expander("Show raw backend response"):
    st.json(training_data)

st.divider()

st.caption(
    "FedMed academic research prototype. The current demonstration uses "
    "synthetic MRI segmentation data. It is not intended for clinical diagnosis."
)
