import streamlit as st

from app.utils.grpc_client import (
    get_all_hospitals,
    get_hospital_status,
    health_check,
    register_hospital,
)


st.set_page_config(
    page_title="FedMed",
    page_icon="🧠",
    layout="wide",
)


def main():
    st.sidebar.title("🧠 FedMed")
    st.sidebar.caption("Cross-Silo Federated Learning Engine")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Hospital Registration",
            "Federated Learning",
            "Model Training",
        ],
    )

    if page == "Dashboard":
        show_dashboard()

    elif page == "Hospital Registration":
        show_hospital_registration()

    elif page == "Federated Learning":
        show_federated_learning()

    elif page == "Model Training":
        show_model_training()


def show_dashboard():
    st.title("🧠 FedMed Dashboard")
    st.subheader("Registered Hospitals")

    if st.button("Refresh Hospital List"):
        st.rerun()

    with st.spinner("Loading hospitals..."):
        result = get_all_hospitals()

    if not result["success"]:
        st.error(
            f"Failed to load hospitals: "
            f"{result.get('status', 'UNKNOWN')}"
        )
        st.caption(
            result.get(
                "message",
                "Unable to retrieve hospitals from backend.",
            )
        )
        return

    hospitals = result.get("hospitals", [])

    if not hospitals:
        st.info(
            "No hospitals are currently registered "
            "with the FedMed network."
        )
        return

    st.success(
        f"{len(hospitals)} hospital(s) registered"
    )

    columns = st.columns(3)

    for index, hospital in enumerate(hospitals):
        with columns[index % 3]:
            with st.container(border=True):
                st.subheader(
                    hospital["hospital_name"]
                )

                st.write(
                    f"**Hospital ID:** "
                    f"{hospital['hospital_id']}"
                )

                st.write(
                    f"**Location:** "
                    f"{hospital['location']}"
                )

                status = hospital["status"]

                if status.lower() == "online":
                    st.success(
                        f"Status: {status}"
                    )
                elif status.lower() == "offline":
                    st.error(
                        f"Status: {status}"
                    )
                else:
                    st.warning(
                        f"Status: {status}"
                    )

    st.write(
        "Privacy-preserving collaborative machine learning "
        "for healthcare institutions."
    )

    st.divider()

    st.subheader("Backend Connection")

    if st.button("Check Backend Health"):
        result = health_check()

        if result["success"]:
            st.success(
                f"Backend connected: {result['message']}"
            )
        else:
            st.error(
                f"Backend connection failed: "
                f"{result['status']} - {result['message']}"
            )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Registered Hospitals",
            "Backend Managed",
        )

    with col2:
        st.metric(
            "Training Rounds",
            "0",
        )

    with col3:
        st.metric(
            "Model Status",
            "Not Started",
        )

    st.divider()

    st.info(
        "FedMed enables hospitals to collaboratively train "
        "machine learning models without sharing raw patient data."
    )

    st.subheader("Hospital Status")

    hospital_id = st.text_input(
        "Enter Hospital ID",
        placeholder="hospital-001",
        key="dashboard_hospital_id",
    )

    if st.button("Check Hospital Status"):
        if not hospital_id.strip():
            st.warning("Please enter a Hospital ID.")
            return

        result = get_hospital_status(
            hospital_id.strip()
        )

        if result["success"]:
            st.success("Hospital found")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Hospital ID**")
                st.write(result["hospital_id"])

            with col2:
                st.write("**Hospital Name**")
                st.write(result["hospital_name"])

            with col3:
                st.write("**Location**")
                st.write(result["location"])

            st.success(
                f"Status: {result['status']}"
            )

            st.info(result["message"])

        else:
            if result["status"] == "NOT_FOUND":
                st.error(
                    f"Hospital '{hospital_id}' is not registered."
                )
            else:
                st.error(
                    f"Status check failed: "
                    f"{result['status']}"
                )

            st.caption(result["message"])


def show_hospital_registration():
    st.title("🏥 Hospital Registration")

    st.write(
        "Register a hospital with the FedMed "
        "federated learning network."
    )

    st.divider()

    with st.form("hospital_registration_form"):

        hospital_id = st.text_input(
            "Hospital ID",
            placeholder="hospital-001",
        )

        hospital_name = st.text_input(
            "Hospital Name",
            placeholder="Example Medical Center",
        )

        location = st.text_input(
            "Location",
            placeholder="Hyderabad",
        )

        submitted = st.form_submit_button(
            "Register Hospital"
        )

    if submitted:

        hospital_id = hospital_id.strip()
        hospital_name = hospital_name.strip()
        location = location.strip()

        if not hospital_id:
            st.error("Hospital ID is required.")
            return

        if not hospital_name:
            st.error("Hospital name is required.")
            return

        if not location:
            st.error("Hospital location is required.")
            return

        with st.spinner("Registering hospital..."):

            result = register_hospital(
                hospital_id,
                hospital_name,
                location,
            )

        if result["success"]:

            st.success(
                result["message"]
            )

            st.info(
                "The hospital is now registered "
                "with the FedMed backend."
            )

        else:

            status = result.get("status", "")

            if status == "ALREADY_EXISTS":
                st.warning(
                    f"Hospital '{hospital_id}' "
                    "is already registered."
                )

            elif status == "INVALID_ARGUMENT":
                st.error(
                    "Invalid hospital registration details."
                )

            elif status:
                st.error(
                    f"Registration failed: {status}"
                )

            else:
                st.error(
                    f"Registration failed: "
                    f"{result.get('message', 'Unknown error')}"
                )

            st.caption(
                result.get(
                    "message",
                    "No additional details available."
                )
            )
def show_federated_learning():
    st.title("🔄 Federated Learning")

    st.write(
        "Monitor the FedMed federated learning server "
        "and participating hospitals."
    )

    st.divider()

    # Federated learning server configuration
    st.subheader("Flower Server")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Server Address",
            "localhost:8080",
        )

    with col2:
        st.metric(
            "Configured Rounds",
            "1",
        )

    with col3:
        st.metric(
            "Strategy",
            "FedAvg",
        )

    st.divider()

    # Server status
    st.subheader("Server Status")

    if st.button("Check Flower Server"):

        import socket

        try:
            connection = socket.create_connection(
                ("localhost", 8080),
                timeout=2,
            )
            connection.close()

            st.success(
                "🟢 Flower server is running"
            )

        except OSError:
            st.error(
                "🔴 Flower server is not running"
            )

            st.caption(
                "Start the Flower server from the backend "
                "before checking its status."
            )

    st.divider()

    # Participating hospitals
    st.subheader("Participating Hospitals")

    with st.spinner("Loading hospitals..."):
        result = get_all_hospitals()

    if not result["success"]:
        st.error(
            f"Unable to load hospitals: "
            f"{result.get('status', 'UNKNOWN')}"
        )
        st.caption(
            result.get(
                "message",
                "Could not retrieve hospitals.",
            )
        )
        return

    hospitals = result.get("hospitals", [])

    if not hospitals:
        st.warning(
            "No hospitals are currently registered."
        )
        return

    st.success(
        f"{len(hospitals)} hospital(s) available "
        "for federated learning."
    )

    columns = st.columns(3)

    for index, hospital in enumerate(hospitals):

        with columns[index % 3]:

            with st.container(border=True):

                st.write(
                    f"### 🏥 "
                    f"{hospital['hospital_name']}"
                )

                st.write(
                    f"**Hospital ID:** "
                    f"{hospital['hospital_id']}"
                )

                st.write(
                    f"**Location:** "
                    f"{hospital['location']}"
                )

                status = hospital["status"]

                if status.lower() == "online":
                    st.success(
                        f"🟢 {status}"
                    )
                elif status.lower() == "offline":
                    st.error(
                        f"🔴 {status}"
                    )
                else:
                    st.warning(
                        f"🟡 {status}"
                    )

    st.divider()

    # Federated learning progress
    st.subheader("Federated Learning Progress")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Round",
            "0 / 1",
        )

    with col2:
        st.metric(
            "Clients Required",
            "3",
        )

    with col3:
        st.metric(
            "Training Status",
            "Not Started",
        )

    st.progress(
        0,
        text="Waiting for federated training to start",
    )

    st.info(
        "The Flower server is configured with the "
        "FedAvg strategy and requires 3 hospitals "
        "for federated training."
    )
def show_model_training():
    st.title("🧠 Model Training")

    st.info(
        "Federated model training controls will be "
        "implemented in the next development phase."
    )

    st.write(
        "Model Status: **Not Started**"
    )


if __name__ == "__main__":
    main()