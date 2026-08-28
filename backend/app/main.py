from fastapi import FastAPI


app = FastAPI(
    title="FedMed Backend",
    description="Cross-Silo Federated Learning Backend",
    version="0.1.0",
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