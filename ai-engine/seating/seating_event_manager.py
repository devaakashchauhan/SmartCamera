import time
from typing import Any, Dict, List, Optional, Set, Tuple

from config import settings
from events.event_processor import EventProcessor
from seating.seat_association import SeatAssociation
from seating.table_manager import TableManager
from tracking.person_tracker import TrackedPerson


class SeatingCandidate:
    """Tracks a person who just sat down, confirming stability before emitting SEATING_STARTED."""
    def __init__(self, track_id: int, table_id: str, first_seen: float):
        self.track_id = track_id
        self.table_id = table_id
        self.first_seen = first_seen
        self.last_seen = first_seen


class ActiveSeatingSession:
    """Represents a confirmed ongoing customer seating session."""
    def __init__(self, track_id: int, table_id: str, start_time: float):
        self.track_id = track_id
        self.table_id = table_id
        self.start_time = start_time
        self.last_seen = start_time
        self.is_in_grace_period = False
        self.grace_start_time: Optional[float] = None

    def get_duration(self, current_time: float) -> float:
        """Returns active duration in seconds."""
        end_t = self.grace_start_time if self.is_in_grace_period and self.grace_start_time else current_time
        return max(0.0, end_t - self.start_time)


class SeatingEventManager:
    """
    Manages customer seating lifecycles:
    - Temporal confirmation before starting sessions (filters noise)
    - Active session duration tracking
    - Grace period handling for temporary occlusion / track loss
    - Seating start and departure event emission via EventProcessor
    - Real-time cafe seating analytics
    """

    def __init__(
        self,
        table_manager: TableManager,
        event_processor: Optional[EventProcessor] = None,
        confirm_seconds: float = settings.SEATING_CONFIRM_SECONDS,
        grace_seconds: float = settings.DEPARTURE_GRACE_SECONDS
    ):
        self.table_manager = table_manager
        self.association_engine = SeatAssociation(table_manager)
        self.event_processor = event_processor or EventProcessor()

        self.confirm_seconds = confirm_seconds
        self.grace_seconds = grace_seconds

        # track_id -> SeatingCandidate (unconfirmed)
        self.candidates: Dict[int, SeatingCandidate] = {}

        # track_id -> ActiveSeatingSession (confirmed)
        self.active_sessions: Dict[int, ActiveSeatingSession] = {}

        # Historical completed session metrics
        self.completed_durations: List[float] = []

    def update(
        self,
        people: List[TrackedPerson],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Processes frame detections, updates candidate & active sessions,
        manages table states, and returns live seating analytics.
        """
        now = time.time()

        # 1. Spatial association: get current track_id -> table_id for SITTING persons
        current_associations = self.association_engine.associate(people, frame_shape)
        currently_associated_ids = set(current_associations.keys())

        # 2. Process currently seated people
        for track_id, table_id in current_associations.items():
            if track_id in self.active_sessions:
                # Active session continues
                session = self.active_sessions[track_id]
                session.last_seen = now
                session.is_in_grace_period = False
                session.grace_start_time = None
            elif track_id in self.candidates:
                # Candidate under observation
                cand = self.candidates[track_id]
                if cand.table_id == table_id:
                    cand.last_seen = now
                    # Check if candidate has been stable for confirm_seconds
                    if (now - cand.first_seen) >= self.confirm_seconds:
                        # Confirm session!
                        self.active_sessions[track_id] = ActiveSeatingSession(
                            track_id=track_id,
                            table_id=table_id,
                            start_time=cand.first_seen
                        )
                        del self.candidates[track_id]
                        # Emit SEATING_STARTED
                        self.event_processor.emit_event(
                            event_type="SEATING_STARTED",
                            table_id=table_id,
                            person_track_id=track_id,
                            start_time=cand.first_seen
                        )
                else:
                    # Switched tables, reset candidate
                    self.candidates[track_id] = SeatingCandidate(track_id, table_id, now)
            else:
                # New candidate
                self.candidates[track_id] = SeatingCandidate(track_id, table_id, now)

        # 3. Clean up candidates who stood up or left before confirmation
        expired_candidates = [
            tid
            for tid, cand in self.candidates.items()
            if tid not in currently_associated_ids and (now - cand.last_seen) > 1.5
        ]
        for tid in expired_candidates:
            del self.candidates[tid]

        # 4. Handle active sessions whose person is not currently detected
        ended_session_ids = []
        for track_id, session in self.active_sessions.items():
            if track_id not in currently_associated_ids:
                if not session.is_in_grace_period:
                    # Enter grace period
                    session.is_in_grace_period = True
                    session.grace_start_time = session.last_seen

                # Check if grace period expired
                time_absent = now - (session.grace_start_time or session.last_seen)
                if time_absent >= self.grace_seconds:
                    # Confirm departure!
                    ended_session_ids.append(track_id)

        # 5. Finalize departed sessions
        for track_id in ended_session_ids:
            session = self.active_sessions[track_id]
            departure_time = session.grace_start_time or session.last_seen
            final_duration = max(0.0, departure_time - session.start_time)

            self.completed_durations.append(final_duration)

            # Emit SEATING_ENDED
            self.event_processor.emit_event(
                event_type="SEATING_ENDED",
                table_id=session.table_id,
                person_track_id=track_id,
                start_time=session.start_time,
                end_time=departure_time,
                duration_seconds=final_duration
            )
            del self.active_sessions[track_id]

        # 6. Update TableManager occupancy states & active durations
        table_occupants: Dict[str, List[int]] = {t: [] for t in self.table_manager.tables}
        table_durations: Dict[str, float] = {t: 0.0 for t in self.table_manager.tables}

        for track_id, session in self.active_sessions.items():
            t_id = session.table_id
            if t_id in table_occupants:
                table_occupants[t_id].append(track_id)
                dur = session.get_duration(now)
                # If multiple people at table, take max duration
                table_durations[t_id] = max(table_durations[t_id], dur)

        for t_id, occupants in table_occupants.items():
            self.table_manager.update_table_occupancy(t_id, occupants, now)

        # 7. Compute aggregate metrics
        avg_completed = (
            sum(self.completed_durations) / len(self.completed_durations)
            if self.completed_durations
            else 0.0
        )
        occupied_tables_count = sum(1 for t in self.table_manager.tables.values() if t.status == "OCCUPIED")
        total_tables = len(self.table_manager.tables)
        utilization_pct = (occupied_tables_count / total_tables * 100.0) if total_tables > 0 else 0.0

        return {
            "active_sessions_count": len(self.active_sessions),
            "occupied_tables_count": occupied_tables_count,
            "available_tables_count": total_tables - occupied_tables_count,
            "total_tables": total_tables,
            "table_utilization_pct": round(utilization_pct, 1),
            "table_durations": table_durations,
            "completed_sessions_count": len(self.completed_durations),
            "avg_seating_duration_sec": round(avg_completed, 1),
        }
