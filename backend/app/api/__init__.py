from fastapi import APIRouter
from .cameras import router as cameras_router
from .events import router as events_router
from .analytics import router as analytics_router
from .dashboard import router as dashboard_router
from .camera_stream import router as camera_stream_router

api_router = APIRouter()
api_router.include_router(cameras_router)
api_router.include_router(events_router)
api_router.include_router(analytics_router)
api_router.include_router(dashboard_router)
api_router.include_router(camera_stream_router)
