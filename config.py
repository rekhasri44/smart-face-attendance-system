import os
from datetime import time
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
ATTENDANCE_CSV = "attendance_log.csv"
PERMISSIONS_CSV = "permissions_log.csv"
EMBEDDINGS_CACHE = "embeddings_cache.npy"
CONTACTS_CSV = "contacts.csv"

# ── Recognition ────────────────────────────────────────────────────────────
RECOGNITION_THRESHOLD = 0.6
FRAME_SKIP = 1
CONSEC_DETECTS_REQUIRED = 2
MODEL_USED = 'SFace'
ABSENT_TIMEOUT = 3
MIN_FACE_SIZE = 40
MAX_PEOPLE_PER_FRAME = 10
PERMISSIONS_PER_MONTH = 5

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

# ── MediaPipe Settings ─────────────────────────────────────────────────────
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
