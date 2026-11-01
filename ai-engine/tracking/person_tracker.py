import uuid
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from config import settings
from activity.posture_classifier import PostureClassifier


class TrackedPerson(NamedTuple):
    track_id: Optional[int]
    confidence: float
    box: List[float]             # [x1, y1, x2, y2]
    keypoints: np.ndarray        # (17, 2)
    keypoint_conf: np.ndarray   # (17,)
    posture: str                 # SITTING, STANDING, UNKNOWN


class PersonTracker:
    """
    Person tracker with YOLO11 Pose and ByteTrack.
    Extracts person tracks, poses, and coordinates with tracker session isolation.
    """

    # COCO Keypoint Skeleton pairs
    SKELETON_CONNECTIONS = [
        # Face
        (0, 1), (0, 2), (1, 3), (2, 4),
        # Torso
        (5, 6), (5, 11), (6, 12), (11, 12),
        # Arms
        (5, 7), (7, 9), (6, 8), (8, 10),
        # Legs
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    POSTURE_COLORS = {
        "SITTING": (46, 204, 113),    # Emerald green
        "STANDING": (52, 152, 219),   # Bright sky blue
        "UNKNOWN": (149, 165, 166),   # Concrete gray
    }

    def __init__(
        self,
        model_path: str = settings.POSE_MODEL_PATH,
        tracker_config: str = settings.TRACKER_CONFIG_PATH
    ):
        self.session_id = str(uuid.uuid4())
        print(f"[PersonTracker] Session ID: {self.session_id}")
        print(f"[PersonTracker] Loading Pose Model: {model_path} on {settings.DEVICE}...")
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.posture_classifier = PostureClassifier()

    def track(
        self,
        frame: np.ndarray,
        chair_boxes: Optional[List[List[float]]] = None
    ) -> List[TrackedPerson]:
        """
        Runs tracking & pose estimation, classifies posture, and returns tracked persons.
        """
        results = self.model.track(
            source=frame,
            device=settings.DEVICE,
            imgsz=640,
            conf=settings.POSE_CONF_THRESHOLD,
            persist=True,
            tracker=self.tracker_config,
            verbose=False
        )

        tracked_people: List[TrackedPerson] = []
        if not results or len(results) == 0:
            return tracked_people

        result = results[0]
        boxes = result.boxes
        keypoints_obj = result.keypoints

        if boxes is None or len(boxes) == 0 or keypoints_obj is None:
            return tracked_people

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(boxes)

        kpts_xy = keypoints_obj.xy.cpu().numpy()
        kpts_conf = (
            keypoints_obj.conf.cpu().numpy()
            if keypoints_obj.conf is not None
            else np.ones((len(boxes), 17))
        )

        for i in range(len(boxes)):
            tid = int(track_ids[i]) if track_ids[i] is not None else None
            box = [float(b) for b in xyxy[i]]
            kpt = kpts_xy[i]
            kpt_c = kpts_conf[i]

            # Classify posture with multi-level fallbacks & temporal smoothing
            posture = self.posture_classifier.classify(
                track_id=tid,
                keypoints=kpt,
                confidence=kpt_c,
                person_box=box,
                chair_boxes=chair_boxes
            )

            tracked_people.append(
                TrackedPerson(
                    track_id=tid,
                    confidence=float(confidences[i]),
                    box=box,
                    keypoints=kpt,
                    keypoint_conf=kpt_c,
                    posture=posture
                )
            )

        return tracked_people

    def draw_people(
        self,
        frame: np.ndarray,
        people: List[TrackedPerson],
        time_records: Optional[Dict[int, Any]] = None
    ) -> np.ndarray:
        """
        Draws person bounding boxes, posture banners, skeleton lines,
        and real-time time spent sitting vs standing.
        """
        output = frame.copy()

        for person in people:
            x1, y1, x2, y2 = map(int, person.box)
            posture_color = self.POSTURE_COLORS.get(person.posture, (150, 150, 150))
            is_tracked = person.track_id is not None

            # Bounding box color: Cyan/Blue for tracked, Orange for untracked
            box_color = (255, 180, 0) if is_tracked else (0, 165, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)

            # Draw Skeleton
            kpts = person.keypoints
            kconf = person.keypoint_conf
            min_conf = settings.KEYPOINT_CONF_THRESHOLD

            for p1_idx, p2_idx in self.SKELETON_CONNECTIONS:
                if kconf[p1_idx] >= min_conf and kconf[p2_idx] >= min_conf:
                    pt1 = (int(kpts[p1_idx][0]), int(kpts[p1_idx][1]))
                    pt2 = (int(kpts[p2_idx][0]), int(kpts[p2_idx][1]))
                    cv2.line(output, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

            # Draw Keypoint joints
            for idx in range(len(kpts)):
                if kconf[idx] >= min_conf:
                    pt = (int(kpts[idx][0]), int(kpts[idx][1]))
                    cv2.circle(output, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)

            # Header Banner: ID + Posture + Sitting/Standing Duration
            id_text = f"ID {person.track_id}" if is_tracked else "Untracked"
            label = f"{id_text}: {person.posture}"

            # Append live time spent sitting vs standing
            rec = time_records.get(person.track_id) if (time_records and is_tracked) else None
            if rec:
                sit_s = int(getattr(rec, "sitting_seconds", 0))
                std_s = int(getattr(rec, "standing_seconds", 0))
                sit_str = f"{sit_s // 60}m {sit_s % 60}s" if sit_s >= 60 else f"{sit_s}s"
                std_str = f"{std_s // 60}m {std_s % 60}s" if std_s >= 60 else f"{std_s}s"
                label += f" [Sit: {sit_str} | Stand: {std_str}]"

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            banner_y1 = max(0, y1 - th - baseline - 8)
            banner_y2 = max(th + baseline + 8, y1)

            # Background rectangle for crisp contrast
            cv2.rectangle(output, (x1, banner_y1), (x1 + tw + 12, banner_y2), posture_color, -1)
            # Text inside banner
            cv2.putText(
                output,
                label,
                (x1 + 6, max(th + 2, y1 - baseline - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return output
