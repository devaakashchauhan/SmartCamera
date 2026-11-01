import math
import time
from collections import deque, Counter
from typing import Dict, List, Optional, Tuple
import numpy as np

from config import settings


class PostureClassifier:
    """
    Robust posture classification (SITTING, STANDING, UNKNOWN) with:
    1. Full-leg knee angles (Hip-Knee-Ankle)
    2. Torso-thigh angles (Shoulder-Hip-Knee)
    3. Spatial fallbacks: Chair overlap (IoU) & Bounding box aspect ratio
    4. Temporal smoothing (majority voting over recent frames)
    """

    def __init__(self, window_size: int = settings.TEMPORAL_WINDOW_SIZE):
        self.window_size = window_size
        # History map: track_id -> deque of recent classifications
        self.history: Dict[int, deque] = {}
        self.last_seen: Dict[int, float] = {}

    def calculate_angle(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float]
    ) -> Optional[float]:
        """
        Calculates the interior angle at joint B given three points A, B, C.
        """
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.hypot(*ba)
        mag_bc = math.hypot(*bc)

        if mag_ba == 0 or mag_bc == 0:
            return None

        cosine = dot / (mag_ba * mag_bc)
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))

    def compute_box_iou(self, boxA: List[float], boxB: List[float]) -> float:
        """
        Compute Intersection over Union (IoU) or overlap between two bounding boxes [x1, y1, x2, y2].
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h

        if inter_area <= 0:
            return 0.0

        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Overlap with respect to the smaller object (often chair or person torso)
        smaller_area = min(boxA_area, boxB_area)
        if smaller_area <= 0:
            return 0.0

        return inter_area / smaller_area

    def check_chair_interaction(
        self,
        person_box: List[float],
        chair_boxes: List[List[float]]
    ) -> bool:
        """
        Checks if person bounding box overlaps significantly with any detected chair.
        """
        if not chair_boxes or person_box is None:
            return False

        for chair_box in chair_boxes:
            overlap = self.compute_box_iou(person_box, chair_box)
            if overlap >= settings.CHAIR_IOU_THRESHOLD:
                return True
        return False

    def classify_frame(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        person_box: Optional[List[float]] = None,
        chair_boxes: Optional[List[List[float]]] = None
    ) -> str:
        """
        Hierarchical posture evaluation for a single frame:
        1. Knee angle (full leg visible)
        2. Torso angle (Shoulder - Hip - Knee)
        3. Seated-at-desk / Webcam fallback (chair overlap + aspect ratio)
        """
        conf_thresh = settings.KEYPOINT_CONF_THRESHOLD

        # COCO keypoint indices:
        # 5: L shoulder, 6: R shoulder
        # 11: L hip,     12: R hip
        # 13: L knee,    14: R knee
        # 15: L ankle,   16: R ankle

        # --- LEVEL 1: Full-Leg Knee Angles (Hip - Knee - Ankle) ---
        knee_angles = []
        for hip, knee, ankle in [(11, 13, 15), (12, 14, 16)]:
            if (
                confidence[hip] >= conf_thresh
                and confidence[knee] >= conf_thresh
                and confidence[ankle] >= conf_thresh
            ):
                ang = self.calculate_angle(keypoints[hip], keypoints[knee], keypoints[ankle])
                if ang is not None:
                    knee_angles.append(ang)

        if knee_angles:
            avg_knee = sum(knee_angles) / len(knee_angles)
            if avg_knee <= settings.KNEE_ANGLE_SITTING_MAX:
                return "SITTING"
            elif avg_knee >= settings.KNEE_ANGLE_STANDING_MIN:
                return "STANDING"

        # --- LEVEL 2: Torso-Thigh Angles (Shoulder - Hip - Knee) ---
        # Useful when ankles are occluded (under table/desk) but hips and knees are visible
        torso_angles = []
        for shoulder, hip, knee in [(5, 11, 13), (6, 12, 14)]:
            if (
                confidence[shoulder] >= conf_thresh
                and confidence[hip] >= conf_thresh
                and confidence[knee] >= conf_thresh
            ):
                ang = self.calculate_angle(keypoints[shoulder], keypoints[hip], keypoints[knee])
                if ang is not None:
                    torso_angles.append(ang)

        if torso_angles:
            avg_torso = sum(torso_angles) / len(torso_angles)
            if avg_torso <= settings.TORSO_ANGLE_SITTING_MAX:
                return "SITTING"
            elif avg_torso >= settings.TORSO_ANGLE_STANDING_MIN:
                return "STANDING"

        # --- LEVEL 3: Spatial & Framing Fallbacks (Webcam / Desk / Partial Occlusion) ---
        if person_box is not None:
            bw = max(1.0, person_box[2] - person_box[0])
            bh = max(1.0, person_box[3] - person_box[1])
            aspect_ratio = bh / bw  # Height / Width

            in_chair = self.check_chair_interaction(person_box, chair_boxes or [])

            # Case A: Person overlaps with a detected chair
            if in_chair:
                # If interacting with a chair and not towering over it with full standing ratio
                if aspect_ratio <= settings.ASPECT_RATIO_STANDING_MIN:
                    return "SITTING"

            # Case B: Upper-body framing (cropped at desk/waist)
            # When seated at a desk, the bounding box is short & square/wide
            if aspect_ratio <= settings.ASPECT_RATIO_SITTING_MAX:
                # Check if shoulders and head are visible and upright
                if confidence[5] >= conf_thresh and confidence[6] >= conf_thresh:
                    # Shoulders are roughly horizontal and hips (if detected) are below
                    return "SITTING"

            # Case C: Person is standing in the room (tall upright box)
            # Even if ankles/lower legs are blocked by foreground objects (like a water bottle)
            if aspect_ratio >= settings.ASPECT_RATIO_STANDING_MIN:
                if (
                    (confidence[5] >= conf_thresh or confidence[6] >= conf_thresh)
                    and (confidence[11] >= conf_thresh or confidence[12] >= conf_thresh)
                ):
                    return "STANDING"

        return "UNKNOWN"

    def classify(
        self,
        track_id: Optional[int],
        keypoints: np.ndarray,
        confidence: np.ndarray,
        person_box: Optional[List[float]] = None,
        chair_boxes: Optional[List[List[float]]] = None
    ) -> str:
        """
        Classifies posture with temporal smoothing over recent frames for stable tracks.
        """
        raw_posture = self.classify_frame(keypoints, confidence, person_box, chair_boxes)

        now = time.time()
        self.cleanup_expired_tracks(now)

        # Untracked person: return raw posture directly
        if track_id is None or track_id < 0:
            return raw_posture

        # Tracked person: maintain sliding window for majority voting
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.window_size)

        self.last_seen[track_id] = now

        # Only add confident classifications to history to avoid polluting with brief unknown dips
        if raw_posture != "UNKNOWN" or len(self.history[track_id]) == 0:
            self.history[track_id].append(raw_posture)

        # Majority vote
        counts = Counter(self.history[track_id])
        smoothed_posture = counts.most_common(1)[0][0]
        return smoothed_posture

    def cleanup_expired_tracks(self, current_time: float):
        """Remove tracks that haven't been observed recently to prevent memory leaks."""
        expired_ids = [
            tid
            for tid, last_t in self.last_seen.items()
            if (current_time - last_t) > settings.TRACK_EXPIRY_SECONDS
        ]
        for tid in expired_ids:
            self.history.pop(tid, None)
            self.last_seen.pop(tid, None)
