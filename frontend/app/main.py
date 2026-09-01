import streamlit as st

from app.utils.grpc_client import (
    health_check,
    register_hospital,
)


st.set_page_config(
    page_title="FedMed",
    page_icon="🏥",
    layout="wide",
)


def main():
    st.sidebar.title("🏥 FedMed")
    st.sidebar.caption(
        "Cross-Silo Federated Learning Engine"
    )

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
    st.title("🏥 FedMed Dashboard")

    st.subheader(
        "Cross-Silo Federated Learning Engine"
    )

    st.write(
        "Privacy-preserving collaborative machine "
        "learning for healthcare institutions."
    )

    st.divider()

    # Backend health check
    st.subheader("Backend Status")

    health = health_check()

    if health["success"]:
        st.success(
            f"Backend Connected: {health['message']}"
        )
    else:
        st.error(
            f"Backend Error: {health['message']}"
        )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Registered Hospitals",
            "0",
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
        "FedMed enables hospitals to collaboratively "
        "train machine learning models without sharing "
        "raw patient data."
    )

    st.subheader("System Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏥 Hospitals")

        st.write(
            "Hospitals can register with the FedMed "
            "federated learning network."
        )

    with col2:
        st.markdown("### 🔐 Privacy")

        st.write(
            "Raw patient data remains inside each "
            "hospital. Only model updates are shared."
        )


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

            if (
                not hospital_id.strip()
                or not hospital_name.strip()
                or not location.strip()
            ):
                st.error(
                    "Please fill in all fields."
                )

            else:

                with st.spinner(
                    "Registering hospital..."
                ):
                    result = register_hospital(
                        hospital_id.strip(),
                        hospital_name.strip(),
                        location.strip(),
                    )

                if result["success"]:
                    st.success(
                        result["message"]
                    )

                else:
                    st.error(
                        result["message"]
                    )

                # Show backend response
                with st.expander(
                    "Backend Response"
                ):
                    st.write(
                        result
                    )


def show_federated_learning():
    st.title("🔄 Federated Learning")

    st.info(
        "Federated learning configuration will be "
        "implemented in the next development phase."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Connected Hospitals",
            "0",
        )

    with col2:
        st.metric(
            "Current Round",
            "0",
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