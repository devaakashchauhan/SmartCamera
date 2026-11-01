from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ActivityEventCreate(BaseModel):
    camera_id: str = "camera_0"
    person_track_id: int
    event_type: str
    posture: str = "UNKNOWN"
    location: Optional[str] = "MAIN_AREA"
    duration_seconds: Optional[float] = None
    timestamp: Optional[datetime] = None


from pydantic import BaseModel, ConfigDict, Field


class ActivityEventCreate(BaseModel):
    camera_id: str = "camera_0"
    person_track_id: int
    event_type: str
    posture: str = "UNKNOWN"
    location: Optional[str] = "MAIN_AREA"
    duration_seconds: Optional[float] = None
    timestamp: Optional[datetime] = None


class ActivityEventResponse(ActivityEventCreate):
    id: int
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class LiveFrameStats(BaseModel):
    camera_id: str = "camera_0"
    fps: float
    device: str
    tracked_count: int
    untracked_count: int
    seated_count: int
    standing_count: int
    objects_count: int
    system_detected_seconds: float = 0.0
    system_undetected_seconds: float = 0.0
    system_coverage_pct: float = 100.0
    person_time_records: List[Dict[str, Any]] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)
