"""Central configuration for the MLP sign language model (MediaPipe landmarks)."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
# Trained model files live in the shared trained_model/ folder one level up
MODEL_DIR = BASE_DIR.parent / "trained_model"

# ── Classes — 33 labels ───────────────────────────────────────────────────────
ASL_CLASSES: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del"]
EMERGENCY_CLASSES: list[str] = ["help", "danger", "emergency", "thumbs_down", "ok_sign"]
ALL_CLASSES: list[str] = ASL_CLASSES + EMERGENCY_CLASSES  # 33 total

# ── Input / architecture ──────────────────────────────────────────────────────
NUM_LANDMARKS = 21  # MediaPipe hand landmarks
COORDS_PER_LANDMARK = 3  # x, y, z
INPUT_DIM = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
HIDDEN1 = 256
HIDDEN2 = 128
NUM_CLASSES = len(ALL_CLASSES)  # 33
DROPOUT1 = 0.3
DROPOUT2 = 0.2

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.20
RANDOM_SEED = 42
SYNTHETIC_VARIATIONS = 500  # per emergency sign

# ── Kaggle dataset ────────────────────────────────────────────────────────────
KAGGLE_DATASET = "grassknoted/asl-alphabet"
IMAGES_PER_LETTER = 800

# ── Export ────────────────────────────────────────────────────────────────────
PT_MODEL_NAME = "sign_language_mlp.pt"
ONNX_MODEL_NAME = "sign_language_mlp.onnx"
LABELS_NAME = "class_names_mlp.json"
