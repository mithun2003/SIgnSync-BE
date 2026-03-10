"""Central configuration for the PyTorch sign language model."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "trained_model"
SKELETON_DIR = DATA_DIR / "skeleton"
CUSTOM_DATA_DIR = DATA_DIR / "custom"

# ── Classes ──────────────────────────────────────────────────────────────────
ASL_CLASSES: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del"]

# Emergency signs — synthetic landmarks, no webcam collection needed
EMERGENCY_CLASSES: list[str] = ["help", "danger", "emergency", "thumbs_down", "ok_sign"]

ALL_CLASSES: list[str] = ASL_CLASSES + EMERGENCY_CLASSES  # 33 total

# Kaggle dataset (contains A-Z, space, del, nothing)
KAGGLE_DATASET = "grassknoted/asl-alphabet"

# ── Emergency sign gesture instructions (shown during data collection) ───────
EMERGENCY_SIGN_INSTRUCTIONS: dict[str, str] = {
    "help": (
        "Open flat palm — spread ALL 5 fingers wide apart, palm facing camera. "
        "Like a 'stop' gesture with maximum finger spread."
    ),
    "danger": (
        "ILY sign — extend INDEX finger up, PINKY finger up, and THUMB out. "
        "Curl middle and ring fingers into palm. "
        "Like the 'I love you' sign in ASL."
    ),
    "emergency": (
        "Thumbs-up fist — curl all 4 fingers into a fist, "
        "then point your THUMB straight UP (not sideways). "
        "Like a strong thumbs-up signal."
    ),
    "thumbs_down": (
        "Thumbs-down fist — curl all 4 fingers into a fist, "
        "then point your THUMB straight DOWN toward the floor. "
        "Opposite of thumbs-up. Globally understood 'no/danger' signal."
    ),
    "ok_sign": (
        "OK circle — touch your THUMB tip to your INDEX finger tip forming "
        "a circle. Curl MIDDLE, RING, and PINKY fingers DOWN into the palm "
        "(NOT up — that would look like ASL F). "
        "Universal 'okay/all good?' check-in sign."
    ),
}

# ── Image settings ────────────────────────────────────────────────────────────
IMAGE_SIZE = 224

# ── DataLoader settings ───────────────────────────────────────────────────────
BATCH_SIZE = 32
# NOTE: Keep NUM_WORKERS = 0 on Windows to avoid multiprocessing issues.
# On Linux/Mac you can increase this (e.g. 4) for faster data loading.
NUM_WORKERS = 0
VAL_SPLIT = 0.15
RANDOM_SEED = 42

# ── Training phases ───────────────────────────────────────────────────────────
# Phase 1: Classifier head only (backbone frozen)
PHASE1_EPOCHS = 15
PHASE1_LR = 1e-3

# Phase 2: Unfreeze last 2 EfficientNet blocks
PHASE2_EPOCHS = 20
PHASE2_LR = 5e-5

# Phase 3: Unfreeze entire network
PHASE3_EPOCHS = 10
PHASE3_LR = 1e-5

# ── Data collection ───────────────────────────────────────────────────────────
FRAMES_PER_SIGN = 300  # frames to collect per custom emergency sign
CAPTURE_FPS = 20  # approximate capture rate (1/CAPTURE_FPS seconds between frames)
