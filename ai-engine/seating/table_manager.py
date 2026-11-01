from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class CafeTable:
    def __init__(
        self,
        table_id: str,
        name: str,
        polygon: List[Tuple[float, float]],
        capacity: int = 2
    ):
        self.table_id = table_id
        self.name = name
        self.polygon = polygon  # Normalized (0.0 to 1.0) coordinates
        self.capacity = capacity
        self.status = "AVAILABLE"  # AVAILABLE, OCCUPIED, UNKNOWN
        self.occupants: List[int] = []  # List of person track_ids
        self.active_since: Optional[float] = None


class TableManager:
    """
    Manages cafe tables, spatial layouts, occupancy states, and visual boundaries.
    """

    TABLE_COLORS = {
        "AVAILABLE": (46, 204, 113),   # Vibrant Emerald Green
        "OCCUPIED": (41, 128, 185),    # Warm Blue / Amber
        "UNKNOWN": (149, 165, 166),    # Gray
    }

    def __init__(self, custom_tables: Optional[List[CafeTable]] = None):
        # Default layout with 3 realistic cafe tables
        self.tables: Dict[str, CafeTable] = {}

        if custom_tables:
            for t in custom_tables:
                self.tables[t.table_id] = t
        else:
            self._init_default_tables()

    def _init_default_tables(self):
        """Initializes default normalized table boundaries covering typical cafe zones."""
        self.tables["TABLE_01"] = CafeTable(
            table_id="TABLE_01",
            name="Table 1 (Front)",
            polygon=[(0.08, 0.45), (0.45, 0.45), (0.45, 0.95), (0.08, 0.95)],
            capacity=2
        )
        self.tables["TABLE_02"] = CafeTable(
            table_id="TABLE_02",
            name="Table 2 (Center)",
            polygon=[(0.55, 0.45), (0.92, 0.45), (0.92, 0.95), (0.55, 0.95)],
            capacity=4
        )

    def get_pixel_polygon(self, table: CafeTable, width: int, height: int) -> np.ndarray:
        """Converts normalized table polygon to pixel coordinates."""
        return np.array([
            (int(x * width), int(y * height))
            for x, y in table.polygon
        ], dtype=np.int32)

    def is_point_in_table(self, table: CafeTable, point: Tuple[int, int], width: int, height: int) -> bool:
        """Tests whether a pixel point lies inside the table polygon."""
        polygon = self.get_pixel_polygon(table, width, height)
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def is_box_overlapping_table(
        self,
        table: CafeTable,
        box: List[float],
        width: int,
        height: int,
        min_overlap: float = 0.15
    ) -> bool:
        """
        Calculates if person bounding box significantly overlaps the table boundary.
        """
        # Person center & bottom anchor points
        cx = int((box[0] + box[2]) / 2)
        cy = int(box[3] * 0.85 + box[1] * 0.15)
        mid_y = int((box[1] + box[3]) / 2)

        # Check either feet point or center point
        if self.is_point_in_table(table, (cx, cy), width, height):
            return True
        if self.is_point_in_table(table, (cx, mid_y), width, height):
            return True

        return False

    def update_table_occupancy(self, table_id: str, occupants: List[int], current_time: float):
        """Updates the occupancy state of a table."""
        table = self.tables.get(table_id)
        if not table:
            return

        table.occupants = occupants
        if len(occupants) > 0:
            table.status = "OCCUPIED"
            if table.active_since is None:
                table.active_since = current_time
        else:
            table.status = "AVAILABLE"
            table.active_since = None

    def draw_tables(
        self,
        frame: np.ndarray,
        active_durations: Dict[str, float],
        alpha: float = 0.18
    ) -> np.ndarray:
        """
        Draws semi-transparent table boundaries with real-time status banners.
        """
        height, width = frame.shape[:2]
        overlay = frame.copy()

        for table_id, table in self.tables.items():
            polygon = self.get_pixel_polygon(table, width, height)
            color = self.TABLE_COLORS.get(table.status, (150, 150, 150))

            # Filled area
            cv2.fillPoly(overlay, [polygon], color)
            # Outline
            cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)

            # Header info
            duration_sec = active_durations.get(table_id, 0.0)
            mins = int(duration_sec // 60)
            secs = int(duration_sec % 60)
            dur_str = f"{mins}m {secs}s" if table.status == "OCCUPIED" else "Available"

            label = f"{table.name}: {table.status} ({dur_str})"
            if table.occupants:
                label += f" | IDs: {table.occupants}"

            label_x = int(polygon[0][0])
            label_y = int(polygon[0][1])

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(
                frame,
                (label_x, max(0, label_y - th - 8)),
                (label_x + tw + 10, max(th + 8, label_y)),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (label_x + 5, max(th + 2, label_y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
