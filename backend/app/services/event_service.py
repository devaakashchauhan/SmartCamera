from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.models.activity import ActivityEvent
from app.schemas.activity import ActivityEventCreate, LiveFrameStats

# In-memory latest frame state broadcasted by AI engine
latest_live_stats: Optional[LiveFrameStats] = None


class EventService:
    @staticmethod
    def record_activity_event(db: Session, event_in: ActivityEventCreate) -> ActivityEvent:
        """Saves an incoming activity event to the database."""
        db_event = ActivityEvent(
            camera_id=event_in.camera_id,
            person_track_id=event_in.person_track_id,
            event_type=event_in.event_type,
            posture=event_in.posture,
            location=event_in.location or "MAIN_AREA",
            duration_seconds=event_in.duration_seconds,
            timestamp=event_in.timestamp or datetime.utcnow()
        )
        db.add(db_event)
        db.commit()
        db.refresh(db_event)
        return db_event

    @staticmethod
    def get_recent_events(
        db: Session,
        limit: int = 50,
        event_type: Optional[str] = None,
        posture: Optional[str] = None
    ) -> List[ActivityEvent]:
        """Queries recent activity events with optional filtering."""
        query = db.query(ActivityEvent)
        if event_type:
            query = query.filter(ActivityEvent.event_type == event_type)
        if posture:
            query = query.filter(ActivityEvent.posture == posture)
        return query.order_by(ActivityEvent.timestamp.desc()).limit(limit).all()

    @staticmethod
    def set_live_stats(stats: LiveFrameStats):
        """Updates in-memory latest frame statistics."""
        global latest_live_stats
        latest_live_stats = stats

    @staticmethod
    def get_live_stats() -> Optional[LiveFrameStats]:
        """Retrieves the latest frame statistics."""
        global latest_live_stats
        return latest_live_stats
