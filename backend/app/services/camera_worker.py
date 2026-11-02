import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Ensure ai-engine is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
ai_engine_dir = root_dir / "ai-engine"
if str(ai_engine_dir) not in sys.path:
    sys.path.insert(0, str(ai_engine_dir))

import cv2
import numpy as np
import torch

from app.database.session import SessionLocal
from app.models.activity import ActivityEvent
from app.schemas.activity import ActivityEventCreate, LiveFrameStats
from app.services.event_service import EventService

from camera.camera_manager import CameraManager
from config import settings as ai_settings
from detection.object_detector import ObjectDetector
from tracking.person_tracker import PersonTracker
from tracking.person_time_tracker import PersonTimeTracker
from zones.zone_manager import ZoneManager


class CameraWorker:
    """
    Background worker that manages the webcam inference loop,
    encodes live annotated frames for web streaming, and pushes telemetry to the database.
    Supports local webcam capture, external annotated frames, and remote raw webcam frames (e.g. Google Colab).
    """

    _instance: Optional["CameraWorker"] = None

    def __init__(self):
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.camera: Optional[CameraManager] = None
        self.detector: Optional[ObjectDetector] = None
        self.tracker: Optional[PersonTracker] = None
        self.zone_manager: Optional[ZoneManager] = None
        self.time_tracker = PersonTimeTracker(grace_period_seconds=6.0)

        self.latest_frame_bytes: Optional[bytes] = None
        self.last_frame_received_time: float = 0.0
        self.lock = threading.Lock()
        self.fps = 0.0
        is_gpu = (ai_settings.DEVICE == 0 or ai_settings.DEVICE == "cuda" or (isinstance(ai_settings.DEVICE, int) and ai_settings.DEVICE >= 0))
        self.device = "GPU" if (torch.cuda.is_available() and is_gpu) else "CPU"
        self.show_zones = False
        self.show_objects = True

        self.person_last_posture = {}
        self._last_upload_time = 0.0

        # Async raw frame buffering & inference thread
        self.raw_lock = threading.Lock()
        self.pending_raw_bytes: Optional[bytes] = None
        self.raw_frame_event = threading.Event()
        self._async_worker_started = False
        self._frame_count = 0
        self._cached_objects = []
        self._cached_chairs = []

    @classmethod
    def get_instance(cls) -> "CameraWorker":
        if cls._instance is None:
            cls._instance = CameraWorker()
        return cls._instance

    def set_external_frame(self, frame_bytes: bytes):
        """Allows external AI engine (main.py) to stream pre-annotated frames directly into the web stream."""
        with self.lock:
            self.latest_frame_bytes = frame_bytes
            self.last_frame_received_time = time.time()

    def push_raw_frame(self, frame_bytes: bytes):
        """
        Receives raw unannotated JPEG frames asynchronously.
        Buffers immediately (<1ms) to release HTTP request thread and prevent network stop-and-wait stalling.
        """
        with self.raw_lock:
            self.pending_raw_bytes = frame_bytes
        self.raw_frame_event.set()

        if not self._async_worker_started:
            self._async_worker_started = True
            threading.Thread(target=self._async_raw_processor_loop, daemon=True).start()

    def _async_raw_processor_loop(self):
        """Dedicated background thread running GPU inference on the freshest uploaded frame."""
        while True:
            self.raw_frame_event.wait(timeout=1.0)
            self.raw_frame_event.clear()

            frame_bytes = None
            with self.raw_lock:
                frame_bytes = self.pending_raw_bytes
                self.pending_raw_bytes = None

            if frame_bytes:
                self.process_raw_frame(frame_bytes)

    def process_raw_frame(self, frame_bytes: bytes) -> dict:
        """
        Processes an unannotated JPEG frame on Google Colab GPU:
        Runs YOLO + posture tracking, updates dwell time records and live MJPEG stream.
        """
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {}

        if self.detector is None:
            self.detector = ObjectDetector()
        if self.tracker is None:
            self.tracker = PersonTracker()
        if self.zone_manager is None:
            self.zone_manager = ZoneManager()

        now = time.time()
        if self._last_upload_time > 0:
            dt = now - self._last_upload_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self.fps = 0.85 * self.fps + 0.15 * instant_fps
        else:
            self.fps = 15.0
        self._last_upload_time = now

        return self._process_and_annotate(frame, now)

    def _process_and_annotate(self, frame: np.ndarray, now: float) -> dict:
        """Core AI pipeline: detection, tracking, dwell timing, telemetry broadcast, overlay rendering."""
        self._frame_count += 1

        # 1. Detect target objects (cache chairs every 3 frames to double GPU throughput; furniture is static)
        if self._frame_count % 3 == 1 or not self._cached_chairs:
            self._cached_objects = self.detector.detect(frame)
            self._cached_chairs = self.detector.get_boxes_by_class(self._cached_objects, "chair")

        detected_objects = self._cached_objects
        chair_boxes = self._cached_chairs

        # 2. Track people & classify posture
        tracked_people = self.tracker.track(frame, chair_boxes=chair_boxes)
        time_stats = self.time_tracker.update(tracked_people, now)

        # 3. Calculate counts
        tracked_count = sum(1 for p in tracked_people if p.track_id is not None)
        untracked_count = sum(1 for p in tracked_people if p.track_id is None)
        seated_count = sum(1 for p in tracked_people if p.posture == "SITTING")
        standing_count = sum(1 for p in tracked_people if p.posture == "STANDING")
        objects_count = len(detected_objects)

        # 4. Check for posture transitions and persist to SQLite
        for p in tracked_people:
            if p.track_id is not None and p.posture != "UNKNOWN":
                prev = self.person_last_posture.get(p.track_id)
                if prev != p.posture:
                    self.person_last_posture[p.track_id] = p.posture
                    zone = self.zone_manager.get_person_zone(p.box, frame.shape) or "MAIN_AREA"
                    event_type = f"{p.posture}_STARTED"

                    try:
                        with SessionLocal() as db:
                            EventService.record_activity_event(
                                db,
                                ActivityEventCreate(
                                    camera_id="camera_0",
                                    person_track_id=p.track_id,
                                    event_type=event_type,
                                    posture=p.posture,
                                    location=zone
                                )
                            )
                    except Exception as e:
                        print(f"[CameraWorker] DB event log error: {e}")

        # 5. Update in-memory live telemetry
        live_stats = LiveFrameStats(
            camera_id="camera_0",
            fps=round(self.fps, 1),
            device=self.device,
            tracked_count=tracked_count,
            untracked_count=untracked_count,
            seated_count=seated_count,
            standing_count=standing_count,
            objects_count=objects_count,
            system_detected_seconds=time_stats["system_detected_seconds"],
            system_undetected_seconds=time_stats["system_undetected_seconds"],
            system_coverage_pct=time_stats["system_coverage_pct"],
            person_time_records=time_stats["active_people"]
        )
        EventService.set_live_stats(live_stats)

        # 6. Render overlays
        annotated = frame.copy()
        if self.show_zones and self.zone_manager:
            annotated = self.zone_manager.draw_zones(annotated)
        if self.show_objects and self.detector:
            annotated = self.detector.draw_detections(annotated, detected_objects)
        if self.tracker:
            annotated = self.tracker.draw_people(annotated, tracked_people, self.time_tracker.active_records)

        # 7. Compress frame to JPEG for web streaming (quality 70 for low latency)
        ret, jpeg_buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            with self.lock:
                self.latest_frame_bytes = jpeg_buffer.tobytes()
                self.last_frame_received_time = now

        return {
            "tracked_count": tracked_count,
            "seated_count": seated_count,
            "standing_count": standing_count,
            "fps": round(self.fps, 1)
        }

    def start(self, source: int | str = 0) -> bool:
        """Starts the camera inference thread if not already running."""
        # Convert numeric string to int if applicable
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        if self.is_running:
            if self.camera and getattr(self.camera, "raw_source", None) == source:
                return True
            print(f"[CameraWorker] Switching source to {source}...")
            self.stop()

        print(f"[CameraWorker] Initializing camera source {source}...")
        self.camera = CameraManager(source=source)
        if not self.camera.connect():
            print("[CameraWorker] Warning: Camera source could not be opened.")
            # If external engine is feeding frames, return true
            if (time.time() - self.last_frame_received_time) < 3.0:
                print("[CameraWorker] External engine is actively feeding frames.")
                return True
            return False

        if self.detector is None:
            self.detector = ObjectDetector()
        if self.tracker is None:
            self.tracker = PersonTracker()
        if self.zone_manager is None:
            self.zone_manager = ZoneManager()

        self.is_running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("[CameraWorker] Camera inference worker started successfully.")
        return True

    def stop(self):
        """Stops the camera worker and releases resources."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.camera:
            self.camera.release()
            self.camera = None
        print("[CameraWorker] Camera worker stopped.")

    def _run_loop(self):
        while self.is_running and self.camera:
            success, frame = self.camera.read_frame()
            if not success or frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            self.fps = self.camera.fps
            self._process_and_annotate(frame, now)
            time.sleep(0.005)

    def stream_generator(self):
        """Yields MJPEG frames for HTTP streaming."""
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Smart Camera Offline - Click Start",
            (80, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (100, 100, 120),
            2,
            cv2.LINE_AA
        )
        _, placeholder_bytes = cv2.imencode(".jpg", placeholder)
        placeholder_data = placeholder_bytes.tobytes()

        while True:
            now = time.time()
            frame_data = None
            with self.lock:
                if self.latest_frame_bytes and (now - self.last_frame_received_time < 3.5):
                    frame_data = self.latest_frame_bytes

            if frame_data is None:
                frame_data = placeholder_data

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
            )
            time.sleep(0.033)  # ~30 FPS stream
