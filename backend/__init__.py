"""
FedMed Production Backend Application Package.
Exposes RESTful APIs, WebSocket live telemetry feeds, hospital client management,
global model weight encryption endpoints, and federation control orchestration.
"""

from backend.app import create_app

__all__ = ["create_app"]
