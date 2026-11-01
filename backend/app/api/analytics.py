from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/summary")
def get_analytics_summary(db: Session = Depends(get_db)):
    """Returns summarized operational statistics."""
    overview = AnalyticsService.get_dashboard_overview(db)
    return {
        "total_people_today": overview.total_people_today,
        "currently_active": overview.currently_active,
        "currently_seated": overview.currently_seated,
        "currently_standing": overview.currently_standing,
        "avg_duration_seconds": overview.avg_session_duration_sec,
        "total_events_today": overview.total_events_today,
    }


@router.get("/hourly")
def get_hourly_analytics(db: Session = Depends(get_db)):
    """Returns hourly activity trends."""
    overview = AnalyticsService.get_dashboard_overview(db)
    return overview.hourly_trends
