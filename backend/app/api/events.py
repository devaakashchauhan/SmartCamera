from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.activity import ActivityEventCreate, ActivityEventResponse, LiveFrameStats
from app.services.event_service import EventService

router = APIRouter(prefix="/events", tags=["Events"])


@router.post("/activity", response_model=ActivityEventResponse)
def record_event(event_in: ActivityEventCreate, db: Session = Depends(get_db)):
    """Receives and stores an activity or posture event from the AI engine."""
    return EventService.record_activity_event(db, event_in)


@router.get("/activity", response_model=List[ActivityEventResponse])
def get_activity_events(
    limit: int = Query(50, ge=1, le=500),
    event_type: Optional[str] = None,
    posture: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Retrieves recent activity events with optional filtering."""
    return EventService.get_recent_events(db, limit=limit, event_type=event_type, posture=posture)


@router.post("/live-stats")
def post_live_stats(stats: LiveFrameStats):
    """Receives real-time frame telemetry from the running AI camera engine."""
    EventService.set_live_stats(stats)
    return {"status": "ok"}


@router.get("/live-stats", response_model=Optional[LiveFrameStats])
def get_live_stats():
    """Returns the most recent camera frame statistics."""
    return EventService.get_live_stats()
