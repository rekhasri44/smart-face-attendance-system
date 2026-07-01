import os
from datetime import time
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
ATTENDANCE_CSV = "attendance_log.csv"
PERMISSIONS_CSV = "permissions_log.csv"
EMBEDDINGS_CACHE = "embeddings_cache.npy"
CACHE_METADATA = "cache_metadata.json"
CONTACTS_CSV = "contacts.csv"

# ── Database ───────────────────────────────────────────────────────────────
DATABASE_PATH = "attendance.db"
USE_DATABASE = True

# ── Cache Configuration ────────────────────────────────────────────────────
CACHE_VERSION = "1.0"
AUTO_REBUILD_CACHE = True
CACHE_WARNING_DAYS = 30

# ── Recognition ────────────────────────────────────────────────────────────
RECOGNITION_THRESHOLD = 0.6
FRAME_SKIP = 1
CONSEC_DETECTS_REQUIRED = 2
MODEL_USED = 'SFace'
ABSENT_TIMEOUT = 3
MIN_FACE_SIZE = 40
MAX_PEOPLE_PER_FRAME = 10
PERMISSIONS_PER_MONTH = 5

# ── Review Queue Configuration ────────────────────────────────────────────
MIN_CONFIDENCE_FOR_REVIEW = 0.45
HIGH_CONFIDENCE = 0.65
REVIEW_QUEUE_ENABLED = True
MAX_REVIEW_ATTEMPTS = 3

# ── Email credentials (loaded from .env) ──────────────────────────────────
SENDER_EMAIL = os.getenv("EMAIL", "")
SENDER_PASSWORD = os.getenv("PASSWORD", "")

# ── Session time windows ──────────────────────────────────────────────────
MORNING_EARLY_START      = time(7, 0)
MORNING_EARLY_END        = time(8, 0)
MORNING_NORMAL_END       = time(9, 0)
MORNING_LATE_END         = time(10, 30)
MORNING_PERMISSION_START = time(10, 30)
MORNING_PERMISSION_END   = time(12, 0)

AFTERNOON_NORMAL_START   = time(13, 0)
AFTERNOON_NORMAL_END     = time(14, 0)
AFTERNOON_LATE_END       = time(14, 30)
AFTERNOON_PERMISSION_START = time(14, 30)
AFTERNOON_PERMISSION_END   = time(15, 0)

QUIT_TIME_START = time(16, 0)
QUIT_TIME_END   = time(20, 0)

# ── Liveness Detection ─────────────────────────────────────────────────────
LIVENESS_ENABLED = True
EAR_THRESHOLD = 0.23
BLINK_CONSECUTIVE_FRAMES = 3
REQUIRED_BLINKS = 1
LIVENESS_DETECTION_WINDOW = 10

# ── Anti-Spoofing Challenge Settings ──────────────────────────────────────
ENABLE_CHALLENGES = True
HEAD_MOVEMENT_THRESHOLD = 0.03
CHALLENGE_TIMEOUT = 5
CHALLENGES_REQUIRED = 2
MAX_CHALLENGE_ATTEMPTS = 3

# ── MediaPipe Settings ─────────────────────────────────────────────────────
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# ── Multi-Camera Configuration ─────────────────────────────────────────────
CAMERA_CONFIG = {
    'main': {'index': 0, 'name': 'Main Entrance'},
    'side': {'index': 1, 'name': 'Side Entrance'},
    # Add more cameras as needed: 'camera2': {'index': 2, 'name': 'Back Door'}
}

ACTIVE_CAMERAS = ['main']  # Which cameras to use
CAMERA_FRAME_SKIP = 2      # Process every Nth frame per camera
CAMERA_SWITCH_INTERVAL = 5 # Seconds between camera switching
