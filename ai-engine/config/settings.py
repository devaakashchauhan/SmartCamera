# Smart Camera - Configuration
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

# Hardware / Device
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Model Paths
OBJECT_MODEL_PATH = str(BASE_DIR / "yolo11n.pt")
POSE_MODEL_PATH = str(BASE_DIR / "yolo11n-pose.pt")
TRACKER_CONFIG_PATH = str(BASE_DIR / "tracking" / "tracker.yaml")

# Target Object Detection Classes
TARGET_CLASSES = [
    "bottle",
    "cup",
    "chair",
    "dining table",
    "laptop",
    "cell phone"
]

# Detection Thresholds
OBJECT_CONF_THRESHOLD = 0.35
POSE_CONF_THRESHOLD = 0.35
KEYPOINT_CONF_THRESHOLD = 0.30

# Posture Classification Thresholds
KNEE_ANGLE_SITTING_MAX = 125.0
KNEE_ANGLE_STANDING_MIN = 150.0

TORSO_ANGLE_SITTING_MAX = 125.0
TORSO_ANGLE_STANDING_MIN = 155.0

# Fallback Posture Thresholds (Webcam / Upper-body framing)
CHAIR_IOU_THRESHOLD = 0.10          # Overlap ratio between person and chair bbox
ASPECT_RATIO_SITTING_MAX = 1.80     # Height / Width ratio (seated at desk/upper-body is usually < 1.75)
ASPECT_RATIO_STANDING_MIN = 2.10    # Full standing body is usually > 2.1

# Temporal Smoothing
TEMPORAL_WINDOW_SIZE = 12           # Number of frames for majority voting
TRACK_EXPIRY_SECONDS = 5.0          # Clean up lost tracks after this duration

# Activity Event Settings
SEATING_CONFIRM_SECONDS = 3.0       # Minimum duration seated to confirm sitting event
DEPARTURE_GRACE_SECONDS = 6.0       # Grace period for temporary occlusion
EVENTS_LOG_PATH = str(BASE_DIR.parent / "storage" / "camera_events.jsonl")
