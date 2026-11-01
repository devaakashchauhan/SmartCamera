import asyncio
import json
from typing import List
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.database.session import get_db, SessionLocal
from app.schemas.dashboard import DashboardOverviewResponse
from app.services.analytics_service import AnalyticsService
from app.services.event_service import EventService

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


class ConnectionManager:
    """Manages active WebSocket connections for live dashboard telemetry."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, data: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception:
                self.disconnect(connection)


ws_manager = ConnectionManager()


@router.get("/overview", response_model=DashboardOverviewResponse)
def get_dashboard_overview(db: Session = Depends(get_db)):
    """Provides high-level monitoring metrics, activity totals, and recent events."""
    return AnalyticsService.get_dashboard_overview(db)


@router.websocket("/ws/live")
async def websocket_live_feed(websocket: WebSocket):
    """
    WebSocket endpoint streaming live telemetry (FPS, people count, postures) to the dashboard.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Send latest stats every 500ms
            live_stats = EventService.get_live_stats()
            payload = live_stats.model_dump(mode="json") if live_stats else {
                "status": "waiting_for_engine",
                "tracked_count": 0,
                "seated_count": 0,
                "standing_count": 0,
                "fps": 0.0
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
