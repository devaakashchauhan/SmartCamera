import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import requests

from config import settings


class EventProcessor:
    """
    Handles event persistence, logging, in-memory caching, and asynchronous backend sync
    for Smart Camera without blocking the real-time video processing loop.
    """

    def __init__(
        self,
        log_path: str = settings.EVENTS_LOG_PATH,
        max_recent: int = 50,
        api_url: str = "http://127.0.0.1:8000/api"
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_recent = max_recent
        self.api_url = api_url
        self.recent_events: List[Dict[str, Any]] = []

    def _sync_to_backend(self, endpoint: str, payload: Dict[str, Any]):
        """Background worker to send events without blocking OpenCV."""
        def _post():
            try:
                requests.post(f"{self.api_url}/{endpoint}", json=payload, timeout=0.5)
            except Exception:
                pass  # Gracefully ignore if backend server is not running

        t = threading.Thread(target=_post, daemon=True)
        t.start()

    def send_live_telemetry(
        self,
        fps: float,
        device: str,
        tracked_count: int,
        untracked_count: int,
        seated_count: int,
        standing_count: int,
        objects_count: int,
        system_detected_seconds: float = 0.0,
        system_undetected_seconds: float = 0.0,
        system_coverage_pct: float = 100.0,
        person_time_records: Optional[List[Dict[str, Any]]] = None,
        camera_id: str = "camera_0"
    ):
        """Pushes live frame telemetry and dwell time analytics to backend asynchronously."""
        payload = {
            "camera_id": camera_id,
            "fps": round(fps, 1),
            "device": device,
            "tracked_count": tracked_count,
            "untracked_count": untracked_count,
            "seated_count": seated_count,
            "standing_count": standing_count,
            "objects_count": objects_count,
            "system_detected_seconds": round(system_detected_seconds, 1),
            "system_undetected_seconds": round(system_undetected_seconds, 1),
            "system_coverage_pct": round(system_coverage_pct, 1),
            "person_time_records": person_time_records or [],
            "timestamp": datetime.utcnow().isoformat()
        }
        self._sync_to_backend("events/live-stats", payload)

    def send_frame_to_stream(self, frame):
        """Streams annotated JPEG frames to backend without blocking."""
        now = time.time()
        if not hasattr(self, "_last_stream_time"):
            self._last_stream_time = 0.0
        if (now - self._last_stream_time) < 0.04:  # Max 25 FPS upload
            return
        self._last_stream_time = now

        def _upload():
            try:
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    requests.post(
                        f"{self.api_url}/camera/upload-frame",
                        data=buf.tobytes(),
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=0.25
                    )
            except Exception:
                pass

        threading.Thread(target=_upload, daemon=True).start()

    def emit_event(
        self,
        event_type: str,
        person_track_id: int,
        start_time: float,
        location: Optional[str] = None,
        posture: str = "UNKNOWN",
        end_time: Optional[float] = None,
        duration_seconds: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Creates, persists locally to JSONL, and syncs a structured Smart Camera event to backend.
        """
        now_dt = datetime.now()
        start_iso = datetime.fromtimestamp(start_time).isoformat()
        end_iso = datetime.fromtimestamp(end_time).isoformat() if end_time else None

        event_payload: Dict[str, Any] = {
            "camera_id": "camera_0",
            "event_type": event_type,
            "person_track_id": person_track_id,
            "posture": posture,
            "location": location or "MAIN_AREA",
            "start_time": start_iso,
            "end_time": end_iso,
            "duration_seconds": round(duration_seconds, 1) if duration_seconds is not None else None,
            "timestamp": now_dt.isoformat(),
            "status": "COMPLETED" if end_time is not None else "ACTIVE"
        }

        if extra:
            event_payload.update(extra)

        # 1. In-memory cache
        self.recent_events.append(event_payload)
        if len(self.recent_events) > self.max_recent:
            self.recent_events.pop(0)

        # 2. Persist locally to JSONL
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_payload) + "\n")
        except Exception as e:
            print(f"[EventProcessor] Error writing event to disk: {e}")

        # 3. Asynchronously push to backend API
        self._sync_to_backend("events/activity", event_payload)

        # 4. Console log
        dur_msg = f" (Duration: {event_payload['duration_seconds']}s)" if duration_seconds else ""
        loc_msg = f" at {location}" if location else ""
        print(f"[EVENT] [{event_type}] Person ID {person_track_id}{loc_msg}{dur_msg}")

        return event_payload

    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent events."""
        return self.recent_events[-limit:]
