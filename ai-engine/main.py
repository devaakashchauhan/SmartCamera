import sys
import time
from pathlib import Path

# Add current dir to sys.path so modules import reliably
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import cv2
import numpy as np

from camera.camera_manager import CameraManager
from config import settings
from detection.object_detector import ObjectDetector
from events.event_processor import EventProcessor
from tracking.person_tracker import PersonTracker
from tracking.person_time_tracker import PersonTimeTracker
from zones.zone_manager import ZoneManager


def draw_hud(
    frame: np.ndarray,
    fps: float,
    device: str,
    tracked_count: int,
    untracked_count: int,
    seated_count: int,
    standing_count: int,
    objects_count: int,
    coverage_pct: float,
    show_zones: bool,
    show_objects: bool,
) -> np.ndarray:
    """
    Renders an executive top status bar (Heads-Up Display) with dwell and coverage telemetry.
    """
    height, width = frame.shape[:2]
    hud_h = 52

    # Dark translucent banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, hud_h), (18, 22, 28), -1)
    frame = cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)
    cv2.line(frame, (0, hud_h), (width, hud_h), (70, 85, 105), 1)

    # Brand Title: SMART CAMERA
    cv2.putText(
        frame,
        "SMART CAMERA",
        (16, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )

    unassigned_str = f" ({untracked_count} untracked)" if untracked_count > 0 else ""
    metrics = [
        f"FPS: {fps:.1f}",
        f"Device: {device}",
        f"People: {tracked_count}{unassigned_str}",
        f"Seated: {seated_count} | Standing: {standing_count}",
        f"Objects: {objects_count}",
        f"Detected: {coverage_pct:.0f}%",
        f"[Z] Zones: {'ON' if show_zones else 'OFF'}",
        f"[O] Objects: {'ON' if show_objects else 'OFF'}",
    ]

    start_x = 210
    step_x = max(105, (width - start_x) // len(metrics))

    for i, metric in enumerate(metrics):
        x = start_x + (i * step_x)
        if x + 95 > width:
            break
        cv2.putText(
            frame,
            metric,
            (x, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (225, 230, 235),
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_activity_card(
    frame: np.ndarray,
    tracked_count: int,
    seated_count: int,
    standing_count: int,
    untracked_count: int,
    time_stats: dict,
    active_records: dict,
) -> np.ndarray:
    """
    Renders an in-view activity summary card with person sit/stand times and system detection coverage.
    """
    height, width = frame.shape[:2]
    card_w = 340
    card_h = 175
    x1, y1 = 16, height - card_h - 16
    x2, y2 = x1 + card_w, height - 16

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 24, 30), -1)
    frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 85, 105), 1)

    cv2.putText(
        frame,
        "ACTIVITY & TIME ANALYTICS",
        (x1 + 10, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )

    total_active = tracked_count + untracked_count
    lines = [
        (f"• Active People: {total_active} (Sit: {seated_count} | Stand: {standing_count})", (255, 255, 255))
    ]

    # Show first 1-2 tracked person's sitting vs standing breakdown and exact timestamps
    for tid, rec in list(active_records.items())[:1]:
        p_str = f"• ID #{tid}: Sit {rec.format_time(rec.sitting_seconds)} | Stand {rec.format_time(rec.standing_seconds)}"
        lines.append((p_str, (46, 204, 113) if rec.current_posture == "SITTING" else (52, 152, 219)))
        sat_str = rec.format_clock_time(rec.last_sat_time)
        std_str = rec.format_clock_time(rec.last_stood_time)
        lines.append((f"  Sat at: {sat_str} | Stood at: {std_str}", (200, 215, 230)))
        if rec.undetected_seconds > 0:
            lines.append((f"  Undetected/Lost: {rec.format_time(rec.undetected_seconds)} ({rec.detection_rate_pct:.0f}% seen)", (180, 180, 180)))

    # System-level detection vs undetected (idle) time
    sys_det = time_stats.get("system_detected_formatted", "0s")
    sys_undet = time_stats.get("system_undetected_formatted", "0s")
    cov_pct = time_stats.get("system_coverage_pct", 100.0)
    lines.append((f"• System Detected: {sys_det} ({cov_pct:.0f}%)", (0, 215, 255)))
    lines.append((f"• System Undetected (Idle): {sys_undet}", (150, 150, 160)))

    y_pos = y1 + 42
    for txt, color in lines[:6]:  # Show up to 6 lines cleanly
        cv2.putText(
            frame,
            txt,
            (x1 + 10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            color,
            1,
            cv2.LINE_AA,
        )
        y_pos += 20

    return frame


def main():
    print("=" * 65)
    print(" Smart Camera - Real-time Person, Posture & Dwell Tracking")
    print(f" Target Device: {settings.DEVICE}")
    print(" Hotkeys: [Q] Quit | [Z] Toggle Zones | [O] Toggle Objects")
    print("=" * 65)

    # Initialize modules
    camera = CameraManager(source=0)
    if not camera.connect():
        print("[Main] Failed to open webcam. Exiting.")
        return

    object_detector = ObjectDetector()
    tracker = PersonTracker()
    zone_manager = ZoneManager()
    event_processor = EventProcessor()
    time_tracker = PersonTimeTracker(grace_period_seconds=6.0)

    show_zones = False
    show_objects = True
    last_telemetry_time = 0.0

    person_last_posture = {}

    try:
        while True:
            success, frame = camera.read_frame()
            if not success or frame is None:
                print("[Main] Camera feed lost or interrupted.")
                break

            now = time.time()

            # 1. Detect target objects
            detected_objects = object_detector.detect(frame)
            chair_boxes = object_detector.get_boxes_by_class(detected_objects, "chair")

            # 2. Track people & classify posture
            tracked_people = tracker.track(frame, chair_boxes=chair_boxes)

            # 3. Update Sit vs Stand & Undetected Time Tracker
            time_stats = time_tracker.update(tracked_people, now)

            # 4. Calculate metrics
            tracked_count = sum(1 for p in tracked_people if p.track_id is not None)
            untracked_count = sum(1 for p in tracked_people if p.track_id is None)
            seated_count = sum(1 for p in tracked_people if p.posture == "SITTING")
            standing_count = sum(1 for p in tracked_people if p.posture == "STANDING")
            objects_count = len(detected_objects)

            # 5. Check for posture transitions and emit activity events
            for p in tracked_people:
                if p.track_id is not None and p.posture != "UNKNOWN":
                    prev = person_last_posture.get(p.track_id)
                    if prev != p.posture:
                        person_last_posture[p.track_id] = p.posture
                        zone = zone_manager.get_person_zone(p.box, frame.shape) or "MAIN_AREA"
                        event_type = f"{p.posture}_STARTED"
                        event_processor.emit_event(
                            event_type=event_type,
                            person_track_id=p.track_id,
                            start_time=now,
                            location=zone,
                            posture=p.posture
                        )

            # 6. Push live telemetry & dwell analytics to backend API every 1.0 second
            dev_label = "GPU" if settings.DEVICE == 0 else "CPU"
            if now - last_telemetry_time >= 1.0:
                event_processor.send_live_telemetry(
                    fps=camera.fps,
                    device=dev_label,
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
                last_telemetry_time = now

            # 7. Render visualization layers
            display = frame.copy()

            if show_zones:
                display = zone_manager.draw_zones(display)

            if show_objects:
                display = object_detector.draw_detections(display, detected_objects)

            # Render Tracked People with Sit/Stand time overlays
            display = tracker.draw_people(display, tracked_people, time_tracker.active_records)

            # Render Lower Left Activity & Time Analytics Card
            display = draw_activity_card(
                display,
                tracked_count,
                seated_count,
                standing_count,
                untracked_count,
                time_stats,
                time_tracker.active_records,
            )

            # Render Top HUD with Coverage
            display = draw_hud(
                display,
                camera.fps,
                dev_label,
                tracked_count,
                untracked_count,
                seated_count,
                standing_count,
                objects_count,
                time_stats["system_coverage_pct"],
                show_zones,
                show_objects,
            )

            # 8. Stream frame to Web Dashboard in real time
            event_processor.send_frame_to_stream(display)

            cv2.imshow("Smart Camera - Monitoring", display)

            # Handle interactive keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[Main] User requested exit.")
                break
            elif key == ord("z"):
                show_zones = not show_zones
                print(f"[Main] Zones overlay toggled: {show_zones}")
            elif key == ord("o"):
                show_objects = not show_objects
                print(f"[Main] Objects overlay toggled: {show_objects}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[Main] Smart Camera shutdown complete.")


if __name__ == "__main__":
    main()
