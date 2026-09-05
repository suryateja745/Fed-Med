import streamlit as st

st.set_page_config(
    page_title="FedMed - Hospital Nodes",
    page_icon="H",
    layout="wide",
)

st.title("FedMed Hospital Node Management")
st.caption("Cross-Silo Federated Learning - Mock Hospital Network")

hospitals = [
    {
        "id": "hospital-1",
        "name": "Hospital 1",
        "dataset": "mock-dataset-1",
        "status": "Connected",
    },
    {
        "id": "hospital-2",
        "name": "Hospital 2",
        "dataset": "mock-dataset-2",
        "status": "Connected",
    },
    {
        "id": "hospital-3",
        "name": "Hospital 3",
        "dataset": "mock-dataset-3",
        "status": "Connected",
    },
]

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

st.divider()

st.subheader("Network Summary")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Total Hospitals", len(hospitals))

with c2:
    st.metric(
        "Connected Nodes",
        sum(h["status"] == "Connected" for h in hospitals),
    )

with c3:
    st.metric(
        "Disconnected Nodes",
        sum(h["status"] != "Connected" for h in hospitals),
    )
