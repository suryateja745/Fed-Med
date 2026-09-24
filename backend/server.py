"""
FedMed Backend Server Runner Module.
Provides entry points for ASGI server launching and command-line execution.
"""

from backend.app import app, create_app
from backend.config import get_backend_config

if __name__ == "__main__":
    import uvicorn
    cfg = get_backend_config()
    uvicorn.run("backend.server:app", host=cfg.host, port=cfg.port, reload=True)
