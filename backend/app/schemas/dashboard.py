from datetime import datetime
from typing import Any, Dict, List
from pydantic import BaseModel


class HourlyTraffic(BaseModel):
    hour: str       # e.g. "09:00", "10:00"
    people_count: int
    seated_count: int
    standing_count: int


class DashboardOverviewResponse(BaseModel):
    total_people_today: int
    currently_active: int
    currently_seated: int
    currently_standing: int
    avg_session_duration_sec: float
    total_events_today: int
    active_camera_count: int
    hourly_trends: List[HourlyTraffic]
    recent_events: List[Dict[str, Any]]
