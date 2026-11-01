from datetime import datetime, timedelta
from typing import Dict, List
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.activity import ActivityEvent
from app.models.camera import Camera
from app.schemas.dashboard import DashboardOverviewResponse, HourlyTraffic
from app.services.event_service import EventService


class AnalyticsService:
    @staticmethod
    def get_dashboard_overview(db: Session) -> DashboardOverviewResponse:
        """
        Aggregates operational metrics, current activity, and hourly distributions.
        """
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        # Total unique people observed today
        total_people_today = (
            db.query(func.count(func.distinct(ActivityEvent.person_track_id)))
            .filter(ActivityEvent.timestamp >= today_start)
            .scalar() or 0
        )

        # Total events today
        total_events_today = (
            db.query(func.count(ActivityEvent.id))
            .filter(ActivityEvent.timestamp >= today_start)
            .scalar() or 0
        )

        # Average duration of completed sessions today
        avg_dur = (
            db.query(func.avg(ActivityEvent.duration_seconds))
            .filter(
                ActivityEvent.timestamp >= today_start,
                ActivityEvent.duration_seconds.isnot(None)
            )
            .scalar() or 0.0
        )

        # Count active cameras
        active_cameras = db.query(func.count(Camera.id)).filter(Camera.status == "ONLINE").scalar() or 1

        # Current live stats
        live = EventService.get_live_stats()
        currently_active = (live.tracked_count + live.untracked_count) if live else 0
        currently_seated = live.seated_count if live else 0
        currently_standing = live.standing_count if live else 0

        # Hourly breakdown for today
        hourly_map: Dict[str, Dict[str, int]] = {f"{h:02d}:00": {"people": 0, "seated": 0, "standing": 0} for h in range(24)}
        events_today = (
            db.query(ActivityEvent)
            .filter(ActivityEvent.timestamp >= today_start)
            .all()
        )

        for ev in events_today:
            hr_str = f"{ev.timestamp.hour:02d}:00"
            if hr_str in hourly_map:
                hourly_map[hr_str]["people"] += 1
                if ev.posture == "SITTING":
                    hourly_map[hr_str]["seated"] += 1
                elif ev.posture == "STANDING":
                    hourly_map[hr_str]["standing"] += 1

        # Return the past 12 hours for charting
        current_hour = datetime.utcnow().hour
        selected_hours = [f"{(current_hour - i) % 24:02d}:00" for i in reversed(range(12))]
        hourly_trends = [
            HourlyTraffic(
                hour=h,
                people_count=hourly_map[h]["people"],
                seated_count=hourly_map[h]["seated"],
                standing_count=hourly_map[h]["standing"]
            )
            for h in selected_hours
        ]

        # Recent 10 events
        recent_events = [
            {
                "id": e.id,
                "person_track_id": e.person_track_id,
                "event_type": e.event_type,
                "posture": e.posture,
                "location": e.location,
                "duration_seconds": e.duration_seconds,
                "timestamp": e.timestamp.isoformat()
            }
            for e in EventService.get_recent_events(db, limit=10)
        ]

        return DashboardOverviewResponse(
            total_people_today=total_people_today,
            currently_active=currently_active,
            currently_seated=currently_seated,
            currently_standing=currently_standing,
            avg_session_duration_sec=round(float(avg_dur), 1),
            total_events_today=total_events_today,
            active_camera_count=active_cameras,
            hourly_trends=hourly_trends,
            recent_events=recent_events
        )
