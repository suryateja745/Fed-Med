"""
FedMed Backend Route Blueprints and Endpoints.
"""

from backend.routes.auth import auth_router
from backend.routes.control import router as control_router
from backend.routes.federation import router as federation_router
from backend.routes.hospitals import router as hospitals_router
from backend.routes.models import router as models_router
from backend.routes.security import router as security_router
from backend.routes.websocket import router as websocket_router

__all__ = [
    "auth_router",
    "federation_router",
    "hospitals_router",
    "models_router",
    "security_router",
    "control_router",
    "websocket_router",
]

