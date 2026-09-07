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