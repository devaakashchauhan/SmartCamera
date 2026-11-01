from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String

from app.database.session import Base


class ActivityEvent(Base):
    """Stores discrete activity and posture events emitted by the AI engine."""
    __tablename__ = "activity_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), default="camera_0", index=True)
    person_track_id = Column(Integer, index=True)
    event_type = Column(String(50), nullable=False, index=True)  # SITTING_STARTED, SITTING_ENDED, etc.
    posture = Column(String(20), default="UNKNOWN")              # SITTING, STANDING, UNKNOWN
    location = Column(String(50), default="MAIN_AREA")           # ZONE_A, ZONE_B, etc.
    duration_seconds = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class PersonSession(Base):
    """Tracks continuous observation sessions for a person track."""
    __tablename__ = "person_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), index=True)
    camera_id = Column(String(50), default="camera_0")
    track_id = Column(Integer, index=True)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    primary_posture = Column(String(20), default="UNKNOWN")
    total_duration = Column(Float, default=0.0)
    sitting_duration_seconds = Column(Float, default=0.0)
    standing_duration_seconds = Column(Float, default=0.0)
    undetected_duration_seconds = Column(Float, default=0.0)
    detection_rate_pct = Column(Float, default=100.0)
    status = Column(String(20), default="ACTIVE")  # ACTIVE, COMPLETED
