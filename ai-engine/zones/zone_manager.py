from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class ZoneManager:
    """
    Manages spatial camera zones with normalized coordinate polygons and containment checks.
    """

    ZONE_COLORS = {
        "ZONE_A": (255, 165, 0),   # Orange
        "ZONE_B": (0, 0, 255),     # Red
        "ZONE_C": (0, 255, 255),   # Yellow
    }

    def __init__(self, zones: Optional[Dict[str, List[Tuple[float, float]]]] = None):
        # Normalized coordinates (0.0 to 1.0)
        self.zones = zones or {
            "ZONE_A": [
                (0.05, 0.10),
                (0.45, 0.10),
                (0.45, 0.45),
                (0.05, 0.45)
            ],
            "ZONE_B": [
                (0.55, 0.10),
                (0.95, 0.10),
                (0.95, 0.45),
                (0.55, 0.45)
            ],
            "ZONE_C": [
                (0.05, 0.60),
                (0.95, 0.60),
                (0.95, 0.95),
                (0.05, 0.95)
            ]
        }

    def get_pixel_polygon(self, zone_name: str, width: int, height: int) -> np.ndarray:
        """Convert normalized zone coordinates to integer pixel coordinates."""
        points = self.zones.get(zone_name, [])
        return np.array([
            (int(x * width), int(y * height))
            for x, y in points
        ], dtype=np.int32)

    def get_person_zone(self, person_box: List[float], frame_shape: Tuple[int, int]) -> Optional[str]:
        """
        Determines which zone a person is inside, using the center of their lower torso/feet.
        """
        height, width = frame_shape[:2]
        cx = int((person_box[0] + person_box[2]) / 2)
        cy = int(person_box[3] * 0.9 + person_box[1] * 0.1)

        for zone_name in self.zones:
            polygon = self.get_pixel_polygon(zone_name, width, height)
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                return zone_name
        return None

    def draw_zones(self, frame: np.ndarray, alpha: float = 0.15) -> np.ndarray:
        """
        Renders semi-transparent zone polygons and headers.
        """
        height, width = frame.shape[:2]
        overlay = frame.copy()

        for zone_name in self.zones:
            polygon = self.get_pixel_polygon(zone_name, width, height)
            color = self.ZONE_COLORS.get(zone_name, (255, 255, 0))

            # Filled semi-transparent polygon
            cv2.fillPoly(overlay, [polygon], color)

            # Zone boundary line
            cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)

            # Zone label
            label_x, label_y = polygon[0]
            label = f"{zone_name.replace('_', ' ')}"
            cv2.putText(
                frame,
                label,
                (label_x + 8, max(25, label_y + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

        return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
