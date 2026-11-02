import subprocess
import time
from typing import Optional, Tuple
import cv2
import numpy as np


def resolve_video_source(source: int | str) -> int | str:
    """
    If the source is a YouTube link (regular video or 24/7 Live stream),
    resolves the direct streamable HLS .m3u8 or video URL via yt-dlp.
    Otherwise returns source unmodified.
    """
    if isinstance(source, str) and ("youtube.com" in source or "youtu.be" in source):
        print(f"[CameraManager] Resolving YouTube stream URL: {source}...")
        # 1. Try python yt_dlp library
        try:
            import yt_dlp
            ydl_opts = {
                'format': 'best[height<=720]/best[height<=480]/best',
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
                url = info.get('url')
                if not url and 'formats' in info and info['formats']:
                    url = info['formats'][-1].get('url')
                if url:
                    title = info.get('title', 'YouTube Live Stream')
                    print(f"[CameraManager] Successfully extracted stream: '{title}'")
                    return url
        except Exception as e:
            print(f"[CameraManager] yt_dlp module error: {e}. Trying CLI fallback...")

        # 2. Try CLI fallback
        try:
            cmd = ["yt-dlp", "-f", "best[height<=720]/best[height<=480]/best", "-g", source]
            out = subprocess.check_output(cmd, text=True, timeout=25).strip()
            if out:
                direct_url = out.split("\n")[0].strip()
                print(f"[CameraManager] CLI successfully resolved YouTube URL!")
                return direct_url
        except Exception as e2:
            print(f"[CameraManager] yt-dlp CLI failed: {e2}")

    return source


class CameraManager:
    """
    Manages camera connections (webcam, video files, YouTube Live streams, RTSP)
    with graceful reconnection and FPS calculation.
    """

    def __init__(self, source: int | str = 0, target_width: int = 1280, target_height: int = 720):
        self.raw_source = source
        self.resolved_source = resolve_video_source(source)
        self.source = self.resolved_source
        self.target_width = target_width
        self.target_height = target_height
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.frame_count = 0
        self._consecutive_failures = 0
        self.is_youtube = isinstance(source, str) and ("youtube.com" in source or "youtu.be" in source)

    def connect(self) -> bool:
        """Connect to the camera source and set video properties."""
        print(f"[CameraManager] Connecting to camera source: {self.source}...")
        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            print(f"[CameraManager] Error: Unable to open camera source: {self.source}")
            self.is_connected = False
            return False

        # If it's a live network feed, set minimal buffer to prevent latency accumulation
        try:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass

        actual_w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or self.target_width)
        actual_h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.target_height)
        print(f"[CameraManager] Camera connected! Resolution: {actual_w}x{actual_h}")
        self.is_connected = True
        self.last_frame_time = time.time()
        self._consecutive_failures = 0
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame and update FPS counter."""
        if not self.is_connected or self.capture is None:
            return False, None

        success, frame = self.capture.read()
        if not success or frame is None:
            self._consecutive_failures += 1

            # For pre-recorded video files: loop automatically when reaching the end
            if not self.is_youtube and isinstance(self.raw_source, str) and self._consecutive_failures > 3:
                try:
                    pos = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
                    total = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
                    if total > 0 and pos >= total - 5:
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._consecutive_failures = 0
                        return self.read_frame()
                except Exception:
                    pass

            # For network / live streams, allow temporary packet gaps before disconnecting
            if self._consecutive_failures > 45:  # ~2.5s of missed frames
                print("[CameraManager] Stream disconnected after consecutive frame drops.")
                self.is_connected = False
            return False, None

        self._consecutive_failures = 0
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
