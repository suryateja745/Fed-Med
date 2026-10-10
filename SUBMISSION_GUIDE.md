# FedMed — Submission Guide

## 1. Project Overview

FedMed is a cross-silo federated learning research prototype designed to demonstrate how multiple hospitals can collaborate on model training without transferring their raw medical images to a central training server.

The project focuses on brain-tumor segmentation as a medical imaging use case.

## 2. Technologies

- Python
- FastAPI
- Flower federated learning
- PyTorch and MONAI
- TenSEAL CKKS encryption components
- Streamlit
- Pandas and Requests

## 3. Implemented Features

- FastAPI backend with a health endpoint.
- Three mock hospital clients.
- Flower-based federated training over multiple rounds.
- Global round-loss history.
- Per-hospital training metric persistence.
- Hospital training status reporting.
- Streamlit monitoring dashboard.
- Automated backend test suite.

## 4. Local Verification

Run the backend tests from the backend Python package directory:

```powershell
& ".\.venv313\Scripts\python.exe" -m pytest -q tests
```

Check backend health:

`http://127.0.0.1:8000/health`

Check training status and recorded metrics:

`http://127.0.0.1:8000/api/training/status`

Open the Streamlit dashboard:

`http://localhost:8502`

## 5. Demonstration Results

The verified demonstration completed three federated rounds with three mock hospitals. Global round loss decreased from approximately 0.8752 in round 1 to 0.5288 in round 3. The backend test suite passed 32 tests during the recorded verification.

These results demonstrate the configured prototype workflow, not clinically useful segmentation performance.

## 6. Known Limitations

- The active training demonstration uses synthetic MRI segmentation data.
- Real MRI dataset integration and validation are still required.
- A complete, validated trained-model inference workflow has not been demonstrated.
- The recorded Dice scores are approximately 0.028, so useful segmentation performance has not been established.
- Security settings have not been independently audited.
- Key management, coordinator access to individual updates, transport protection, and formal differential-privacy accounting need further review.
- Some Flower client APIs generate deprecation warnings.

## 7. Conclusion

FedMed demonstrates the core software workflow of a cross-silo federated learning research prototype, including mock hospital clients, federated training rounds, metric persistence, and dashboard monitoring.

Further implementation and evaluation are required before making claims about real medical-image performance, clinical use, or formal privacy guarantees.

**Intended use:** Academic research and demonstration only. Not for clinical diagnosis.
