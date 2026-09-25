"""
FedMed FastAPI Application Factory.
Configures CORS middleware, lifespan event handlers, OpenAPI metadata,
and mounts all REST and WebSocket API routers.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import BackendConfig, get_backend_config
from backend.database import init_db
from backend.dependencies import BackendServices
from backend.routes import (
    auth_router,
    control_router,
    federation_router,
    hospitals_router,
    models_router,
    security_router,
    websocket_router,
)
from federation.utils.logger import setup_logger

logger = setup_logger(name="FedMedApp")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager: sets up services on startup and cleans up on shutdown."""
    cfg = app.state.config
    logger.info("=" * 65)
    logger.info("  Starting FedMed Backend Server")
    logger.info(f"  * Database       : {cfg.database_url}")
    logger.info(f"  * Checkpoint Dir : {cfg.checkpoint_dir.resolve()}")
    logger.info(f"  * Logs & Feeds   : {cfg.logs_dir.resolve()}")
    logger.info(f"  * API Docs       : /docs and /redoc")
    logger.info("=" * 65)

    # Initialize relational database schemas and default seed accounts
    init_db()

    # Initialize backend services singleton
    services = BackendServices.get_instance(cfg)
    app.state.services = services

    # Export initial live dashboard state
    services.api_bridge.export_dashboard_json()

    yield

    logger.info("FedMed Backend Server shutting down cleanly...")


def create_app(config: BackendConfig = None) -> FastAPI:
    """Factory creating and configuring the FedMed FastAPI application."""
    cfg = config or get_backend_config()

    app = FastAPI(
        title=cfg.title,
        version=cfg.version,
        description=cfg.description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.config = cfg

    # Configure CORS Middleware
    origins = cfg.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins != ["*"] else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API Routers
    api_prefix = cfg.api_prefix
    app.include_router(auth_router, prefix=api_prefix)
    app.include_router(federation_router, prefix=api_prefix)
    app.include_router(hospitals_router, prefix=api_prefix)
    app.include_router(models_router, prefix=api_prefix)
    app.include_router(security_router, prefix=api_prefix)
    app.include_router(control_router, prefix=api_prefix)
    app.include_router(websocket_router)

    @app.get("/", tags=["Root"])
    def root():
        """Root endpoint returning API status, documentation links, and project telemetry."""
        return {
            "name": cfg.title,
            "version": cfg.version,
            "status": "ONLINE",
            "docs": "/docs",
            "redoc": "/redoc",
            "api_endpoints": {
                "auth_login": f"{api_prefix}/auth/login",
                "auth_register": f"{api_prefix}/auth/register",
                "auth_me": f"{api_prefix}/auth/me",
                "auth_nodes": f"{api_prefix}/auth/nodes",
                "dashboard": f"{api_prefix}/federation/dashboard",
                "hospitals": f"{api_prefix}/hospitals",
                "models": f"{api_prefix}/models/global",
                "security": f"{api_prefix}/security/status",
                "health": f"{api_prefix}/control/system/health",
                "websocket_telemetry": "/ws/telemetry",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


    @app.get("/health", tags=["Health"])
    def health():
        """Health check endpoint for container orchestrators and load balancers."""
        return {"status": "HEALTHY", "timestamp": datetime.now(timezone.utc).isoformat()}

    return app


# Default application instance for ASGI servers (uvicorn backend.app:app)
app = create_app()
