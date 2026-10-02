"""
Real-Time WebSocket Telemetry Streaming Route for FedMed Frontend Dashboards.
Streams live training progress, hospital heartbeats, convergence curves,
and global checkpoint updates to connected browser clients.
"""

import asyncio
from datetime import datetime, timezone
import json
from typing import Any, Dict
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from backend.dependencies import BackendServices, get_services
from federation.utils.logger import setup_logger

router = APIRouter(tags=["WebSocket"])
logger = setup_logger(name="WebSocketRoute")


@router.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(
    websocket: WebSocket,
    services: BackendServices = Depends(get_services),
):
    """
    WebSocket endpoint streaming live federated learning telemetry directly to UI dashboards.
    Sends full state on connect and streams updates every 2 seconds.
    """
    await services.ws_manager.connect(websocket)

    try:
        # 1. Send initial handshake and current dashboard state
        initial_state = services.api_bridge.get_full_dashboard_state()
        initial_state["participating_hospitals"] = services.get_all_hospitals()
        await websocket.send_json({
            "type": "INITIAL_STATE",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": initial_state,
        })

        # 2. Continuous broadcast loop while connection remains alive
        while True:
            # Check for client messages (e.g. ping)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
                if data == "ping":
                    await websocket.send_json({"type": "PONG", "timestamp": datetime.now(timezone.utc).isoformat()})
            except asyncio.TimeoutError:
                pass

            # Stream periodic telemetry update
            current_state = services.api_bridge.get_full_dashboard_state()
            current_state["participating_hospitals"] = services.get_all_hospitals()
            await websocket.send_json({
                "type": "TELEMETRY_UPDATE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": current_state,
            })

    except WebSocketDisconnect:
        services.ws_manager.disconnect(websocket)
    except Exception as e:
        logger.debug(f"WebSocket closed: {e}")
        services.ws_manager.disconnect(websocket)
