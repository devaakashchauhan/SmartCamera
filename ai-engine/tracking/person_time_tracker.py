import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class PersonTimeRecord:
    """Maintains exact sitting, standing, and undetected/lost duration for a single person,
    including exact timestamps of when they sat down and stood up."""

    def __init__(self, track_id: int, start_time: float, initial_posture: str = "UNKNOWN"):
        self.track_id = track_id
        self.first_seen = start_time
        self.last_seen = start_time
        self.last_update_time = start_time

        self.sitting_seconds = 0.0
        self.standing_seconds = 0.0
        self.undetected_seconds = 0.0

        self.current_posture = initial_posture
        self.current_posture_start_time = start_time

        # Exact timestamps when user sat down or stood up
        self.last_sat_time: Optional[float] = start_time if initial_posture == "SITTING" else None
        self.last_stood_time: Optional[float] = start_time if initial_posture == "STANDING" else None

        self.is_absent = False
        self.absence_start_time = 0.0

    @property
    def total_observed_seconds(self) -> float:
        return self.sitting_seconds + self.standing_seconds

    @property
    def total_session_seconds(self) -> float:
        return self.total_observed_seconds + self.undetected_seconds

    @property
    def detection_rate_pct(self) -> float:
        total = self.total_session_seconds
        return round((self.total_observed_seconds / total * 100.0), 1) if total > 0 else 100.0

    @staticmethod
    def format_clock_time(timestamp: Optional[float]) -> str:
        """Formats unix timestamp as readable 12-hour clock (e.g. 5:48:12 PM)."""
        if not timestamp:
            return "--"
        return datetime.fromtimestamp(timestamp).strftime("%I:%M:%S %p")

    @staticmethod
    def format_time(seconds: float) -> str:
        s = int(seconds)
        mins = s // 60
        secs = s % 60
        return f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "current_posture": self.current_posture,
            "sitting_seconds": round(self.sitting_seconds, 1),
            "standing_seconds": round(self.standing_seconds, 1),
            "undetected_seconds": round(self.undetected_seconds, 1),
            "total_observed_seconds": round(self.total_observed_seconds, 1),
            "total_session_seconds": round(self.total_session_seconds, 1),
            "detection_rate_pct": self.detection_rate_pct,
            "is_absent": self.is_absent,
            "last_sat_time": self.last_sat_time,
            "last_stood_time": self.last_stood_time,
            "last_sat_formatted": self.format_clock_time(self.last_sat_time),
            "last_stood_formatted": self.format_clock_time(self.last_stood_time),
            "sitting_formatted": self.format_time(self.sitting_seconds),
            "standing_formatted": self.format_time(self.standing_seconds),
            "undetected_formatted": self.format_time(self.undetected_seconds),
            "total_formatted": self.format_time(self.total_session_seconds),
        }


class PersonTimeTracker:
    """
    Tracks precise time spent Sitting vs Standing, exact timestamps of when user sits and stands,
    time undetected (system lost / occlusion), and system-wide camera coverage.
    """

    def __init__(self, grace_period_seconds: float = 6.0):
        self.grace_period_seconds = grace_period_seconds
        self.active_records: Dict[int, PersonTimeRecord] = {}
        self.completed_records: List[PersonTimeRecord] = []

        # System-wide camera detection statistics
        self.system_start_time = time.time()
        self.last_system_time = self.system_start_time
        self.system_active_detection_seconds = 0.0   # Time >= 1 person detected
        self.system_idle_undetected_seconds = 0.0    # Time 0 people detected

    def update(
        self,
        tracked_people: List[Any],
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Updates time tracking based on current frame detections.
        """
        now = current_time or time.time()

        # 1. Update system-wide detection vs idle time
        dt_system = min(self.grace_period_seconds, max(0.0, now - self.last_system_time))
        self.last_system_time = now

        visible_tracked_ids = set()
        for p in tracked_people:
            if getattr(p, "track_id", None) is not None:
                visible_tracked_ids.add(p.track_id)

        if len(visible_tracked_ids) > 0:
            self.system_active_detection_seconds += dt_system
        else:
            self.system_idle_undetected_seconds += dt_system

        # 2. Update visible people
        for p in tracked_people:
            tid = getattr(p, "track_id", None)
            if tid is None:
                continue

            posture = getattr(p, "posture", "UNKNOWN")

            if tid not in self.active_records:
                self.active_records[tid] = PersonTimeRecord(tid, now, initial_posture=posture)

            rec = self.active_records[tid]

            # Detect posture transition (when user sits or stands)
            if rec.current_posture != posture and posture in ("SITTING", "STANDING"):
                if posture == "SITTING":
                    rec.last_sat_time = now
                elif posture == "STANDING":
                    rec.last_stood_time = now
                rec.current_posture = posture
                rec.current_posture_start_time = now

            # If person was previously absent (temporary occlusion/track loss), record gap
            if rec.is_absent:
                absent_gap = now - rec.absence_start_time
                rec.undetected_seconds += max(0.0, absent_gap)
                rec.is_absent = False
                rec.last_update_time = now

            # Accumulate posture duration
            dt = min(self.grace_period_seconds, max(0.0, now - rec.last_update_time))
            rec.last_update_time = now
            rec.last_seen = now

            if posture == "SITTING":
                rec.sitting_seconds += dt
            elif posture == "STANDING":
                rec.standing_seconds += dt

        # 3. Handle absent people (in grace period or expired)
        completed_ids = []
        for tid, rec in self.active_records.items():
            if tid not in visible_tracked_ids:
                if not rec.is_absent:
                    rec.is_absent = True
                    rec.absence_start_time = rec.last_seen

                time_absent = now - rec.absence_start_time
                if time_absent >= self.grace_period_seconds:
                    rec.undetected_seconds += time_absent
                    completed_ids.append(tid)

        # 4. Finalize completed tracks
        for tid in completed_ids:
            rec = self.active_records.pop(tid)
            self.completed_records.append(rec)

        # 5. Calculate summary metrics
        total_sys = self.system_active_detection_seconds + self.system_idle_undetected_seconds
        sys_coverage_pct = (
            round((self.system_active_detection_seconds / total_sys * 100.0), 1)
            if total_sys > 0
            else 0.0
        )

        return {
            "active_people": [rec.to_dict() for rec in self.active_records.values()],
            "system_detected_seconds": round(self.system_active_detection_seconds, 1),
            "system_undetected_seconds": round(self.system_idle_undetected_seconds, 1),
            "system_detected_formatted": PersonTimeRecord.format_time(self.system_active_detection_seconds),
            "system_undetected_formatted": PersonTimeRecord.format_time(self.system_idle_undetected_seconds),
            "system_coverage_pct": sys_coverage_pct,
            "completed_count": len(self.completed_records)
        }

    def get_person_record(self, track_id: int) -> Optional[PersonTimeRecord]:
        return self.active_records.get(track_id)
