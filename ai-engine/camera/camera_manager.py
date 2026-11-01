import cv2
import time
from typing import Optional, Tuple
import numpy as np


class CameraManager:
    """
    Manages camera connections (webcam, video files, RTSP streams)
    with graceful reconnection and FPS calculation.
    """

    def __init__(self, source: int | str = 0, target_width: int = 1280, target_height: int = 720):
        self.source = source
        self.target_width = target_width
        self.target_height = target_height
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.frame_count = 0

    def connect(self) -> bool:
        """Connect to the camera source and set video properties."""
        print(f"[CameraManager] Connecting to camera source: {self.source}...")
        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            print(f"[CameraManager] Error: Unable to open camera source: {self.source}")
            self.is_connected = False
            return False

        # Attempt to set frame resolution
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        actual_w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CameraManager] Camera connected! Resolution: {actual_w}x{actual_h}")
        self.is_connected = True
        self.last_frame_time = time.time()
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame and update FPS counter."""
        if not self.is_connected or self.capture is None:
            return False, None

        success, frame = self.capture.read()
        if not success or frame is None:
            self.is_connected = False
            return False, None

        now = time.time()
        dt = now - self.last_frame_time
        if dt > 0:
            current_fps = 1.0 / dt
            # Exponential moving average for smooth FPS display
            self.fps = (0.9 * self.fps + 0.1 * current_fps) if self.fps > 0 else current_fps
        self.last_frame_time = now
        self.frame_count += 1

        return True, frame

    def release(self):
        """Release camera resources."""
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
            print("[CameraManager] Camera released.")
        self.is_connected = False
