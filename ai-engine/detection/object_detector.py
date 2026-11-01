from typing import Dict, List, NamedTuple, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from config import settings


class DetectedObject(NamedTuple):
    label: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]


class ObjectDetector:
    """
    Dedicated Object Detector using YOLO11.
    Detects target objects: bottle, cup, chair, dining table, laptop, cell phone.
    """

    # Distinct color palette (BGR) for detected objects
    CLASS_COLORS = {
        "bottle": (120, 200, 50),      # Lime green
        "cup": (255, 140, 0),          # Amber orange
        "chair": (180, 50, 180),       # Purple
        "dining table": (200, 150, 0), # Teal / Blue-green
        "laptop": (220, 100, 50),      # Blue
        "cell phone": (50, 180, 220),  # Yellow-gold
    }

    def __init__(self, model_path: str = settings.OBJECT_MODEL_PATH):
        print(f"[ObjectDetector] Loading Object Detection Model: {model_path} on {settings.DEVICE}...")
        self.model = YOLO(model_path)
        self.target_classes = settings.TARGET_CLASSES

        self.class_ids = [
            class_id
            for class_id, name in self.model.names.items()
            if name in self.target_classes
        ]
        print(f"[ObjectDetector] Filtered classes: {self.target_classes}")

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Runs object detection on a frame and returns structured detections.
        """
        results = self.model.predict(
            source=frame,
            device=settings.DEVICE,
            imgsz=640,
            conf=settings.OBJECT_CONF_THRESHOLD,
            classes=self.class_ids,
            verbose=False
        )

        detections: List[DetectedObject] = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(xyxy, confidences, cls_ids):
                label = self.model.names.get(cls_id, "object")
                detections.append(
                    DetectedObject(
                        label=label,
                        confidence=float(conf),
                        box=[float(b) for b in box]
                    )
                )

        return detections

    def get_boxes_by_class(self, detections: List[DetectedObject], target_class: str) -> List[List[float]]:
        """Filter bounding boxes by class name."""
        return [d.box for d in detections if d.label.lower() == target_class.lower()]

    def draw_detections(self, frame: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
        """
        Draws object bounding boxes with clean styled labels.
        """
        output = frame.copy()

        for obj in detections:
            x1, y1, x2, y2 = map(int, obj.box)
            color = self.CLASS_COLORS.get(obj.label, (0, 255, 255))

            # Bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Label banner
            label = f"{obj.label} {obj.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                output,
                (x1, max(0, y1 - th - baseline - 4)),
                (x1 + tw + 6, max(th + baseline + 4, y1)),
                color,
                -1
            )
            cv2.putText(
                output,
                label,
                (x1 + 3, max(th, y1 - baseline - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return output
