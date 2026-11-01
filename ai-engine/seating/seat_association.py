from typing import Dict, List, Optional, Tuple
from seating.table_manager import TableManager
from tracking.person_tracker import TrackedPerson


class SeatAssociation:
    """
    Associates tracked people with cafe tables.
    Enforces PRD FR-007: Only persons with confirmed SITTING posture
    overlapping a table zone are associated with that table.
    """

    def __init__(self, table_manager: TableManager):
        self.table_manager = table_manager

    def associate(
        self,
        people: List[TrackedPerson],
        frame_shape: Tuple[int, int]
    ) -> Dict[int, str]:
        """
        Returns a dictionary mapping track_id -> table_id.
        """
        height, width = frame_shape[:2]
        associations: Dict[int, str] = {}

        for person in people:
            # Must be a valid tracked person
            if person.track_id is None:
                continue

            # Must be SITTING to be associated as a seated customer (per FR-007)
            if person.posture != "SITTING":
                continue

            # Check overlap against each table
            for table_id, table in self.table_manager.tables.items():
                if self.table_manager.is_box_overlapping_table(table, person.box, width, height):
                    associations[person.track_id] = table_id
                    break  # Associate with first matching table

        return associations
