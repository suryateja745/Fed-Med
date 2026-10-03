from fastapi import FastAPI

from app.federated.metrics import get_training_state


app = FastAPI(
    title="FedMed Backend",
    description="Cross-Silo Federated Learning Backend",
    version="0.2.0",
)


@app.get("/")
def root():
    return {
        "project": "FedMed",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.get("/api/training/status")
def training_status():
    return get_training_state()

FEDMED_BUILD = "2026-09-26"

# FedMed backend build: 2026-09-28

# FedMed backend build: 2026-09-29

# FedMed backend check: 2026-09-30

# FedMed backend check: 2026-10-01

# FedMed backend check: 2026-10-02

# FedMed backend check: 2026-10-03

# FedMed backend check: 2026-10-03
