"""Sign Language Gesture Prediction — SVM with MediaPipe landmarks.

Primary model: SVM (Scaler → PCA → RBF-SVM) trained on 63-D hand landmarks.
Model file:    src/app/core/ml/trained_model/sign_language_svm.joblib
Labels file:   src/app/core/ml/trained_model/class_names_svm.json

Two prediction paths:
  • predict_sign(image_bytes)           — HTTP endpoint: raw image → MediaPipe → SVM
  • predict_sign_from_landmarks(lms)    — WebSocket: pre-extracted landmarks → SVM
    (frontend already runs MediaPipe; sending 63 floats is 60× cheaper than a JPEG)
"""

import json
import logging
import threading
import urllib.request
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
_SVM_DIR = BASE_DIR / "trained_model"
MODEL_PATH = _SVM_DIR / "sign_language_svm.joblib"
CLASS_PATH = _SVM_DIR / "class_names_svm.json"

CONFIDENCE_THRESHOLD = 85.0  # below this % → return "uncertain" instead of a label

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE  (thread-safe lazy loading)
# ═══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

_pipeline: object | None = None
_class_names: list[str] = []
_model_loaded: bool = False
_model_load_lock = threading.Lock()

# MediaPipe Tasks Vision — HandLandmarker (replaces deprecated mp.solutions.hands)
_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_TASK_MODEL_PATH = _SVM_DIR / "hand_landmarker.task"  # same folder as .joblib

_detector: _mp_vision.HandLandmarker | None = None


def _ensure_task_model() -> str:
    """Download hand_landmarker.task on first use."""
    if not _TASK_MODEL_PATH.exists():
        logger.info("Downloading hand_landmarker.task …")
        urllib.request.urlretrieve(_HAND_LANDMARKER_URL, _TASK_MODEL_PATH)
        logger.info("hand_landmarker.task saved to %s", _TASK_MODEL_PATH)
    return str(_TASK_MODEL_PATH)


def _get_detector() -> _mp_vision.HandLandmarker:
    global _detector
    if _detector is None:
        opts = _mp_vision.HandLandmarkerOptions(
            base_options=_mp_tasks.BaseOptions(model_asset_path=_ensure_task_model()),
            running_mode=_mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
        _detector = _mp_vision.HandLandmarker.create_from_options(opts)
    return _detector


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_ml_model() -> bool:
    """Load SVM pipeline + class names from disk (thread-safe, idempotent)."""
    global _pipeline, _class_names, _model_loaded

    if _model_loaded and _pipeline is not None:
        return True

    with _model_load_lock:
        if _model_loaded and _pipeline is not None:  # double-checked
            return True

        if not MODEL_PATH.exists():
            logger.critical("SVM model not found at %s", MODEL_PATH)
            logger.critical(
                "Train it first:  python src/app/core/ml/sign_model_pytorch/train_svm_standalone.py"
                "  then place sign_language_svm.joblib + class_names_svm.json in src/app/core/ml/trained_model/"
            )
            return False

        if not CLASS_PATH.exists():
            logger.critical("Class names not found at %s", CLASS_PATH)
            return False

        try:
            logger.info("Loading SVM model from %s …", MODEL_PATH)
            _pipeline = joblib.load(MODEL_PATH)
            with open(CLASS_PATH) as f:
                _class_names = json.load(f)
            _model_loaded = True
            logger.info("SVM model loaded — %d classes", len(_class_names))
            return True
        except Exception:
            logger.exception("Failed to load SVM model")
            _model_loaded = False
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# LANDMARK NORMALIZATION  →  82-D feature vector
# ═══════════════════════════════════════════════════════════════════════════════
#
# Feature layout (must match train_svm_standalone.py exactly):
#   [0:63]  — 21 landmarks × (x,y,z), wrist-subtracted + scale-normalised
#   [63:73] — 10 pairwise tip-to-tip distances   (C(5,2) pairs)
#   [73:78] — 5  finger extension scores          (MCP_y − TIP_y)
#   [78:82] — 4  adjacent tip lateral spreads     (TIP_i.x − TIP_{i+1}.x)
#
# Normalization:
#   1. Subtract wrist (landmark 0)   → wrist at (0, 0, 0)
#   2. Divide by wrist-to-middle-MCP → scale-invariant
#   All geometric features are computed from the already-normalised coords.

# Landmark index constants
_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
_MCPS = [2, 5, 9, 13, 17]  # thumb CMC, index MCP, middle MCP, ring MCP, pinky MCP
_TIP_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]  # 10 pairs


def _compute_geometric_features(pts_norm: np.ndarray) -> np.ndarray:
    """Compute 19 geometric features from (21, 3) normalised landmark array.

    These features make similar signs like B vs W easily separable:
    - Tip distances capture whether fingertips are clustered (B) or spread (W)
    - Extension scores distinguish curled vs straight fingers
    - Lateral spreads directly encode the x-spread pattern
    """
    tip_pts = pts_norm[_TIPS]  # (5, 3)
    mcp_pts = pts_norm[_MCPS]  # (5, 3)

    features: list[float] = []

    # 10 pairwise Euclidean distances between the 5 fingertips
    for i, j in _TIP_PAIRS:
        diff = tip_pts[i] - tip_pts[j]
        features.append(float(np.sqrt(np.dot(diff, diff))))

    # 5 finger extension scores: MCP_y − TIP_y  (positive = tip above MCP = extended)
    for i in range(5):
        features.append(float(mcp_pts[i, 1] - tip_pts[i, 1]))

    # 4 adjacent tip x-spreads: TIP_i.x − TIP_{i+1}.x
    for i in range(4):
        features.append(float(tip_pts[i, 0] - tip_pts[i + 1, 0]))

    return np.array(features, dtype=np.float32)  # 19 values


def _normalize_mediapipe_result(lms) -> np.ndarray:
    """Normalize a HandLandmarker result landmark list → (82,) float32."""
    wrist = lms[0]
    coords: list[float] = []
    for lm in lms:
        coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    arr = np.array(coords, dtype=np.float32)
    mid_base = lms[9]
    scale = float(np.sqrt((mid_base.x - wrist.x) ** 2 + (mid_base.y - wrist.y) ** 2))
    if scale > 1e-6:
        arr /= scale
    geo = _compute_geometric_features(arr.reshape(21, 3))
    return np.concatenate([arr, geo])  # 63 + 19 = 82


def _normalize_landmarks_json(landmarks: list[dict]) -> np.ndarray | None:
    """Normalize pre-extracted landmarks from frontend → (82,) float32.

    landmarks: list of 21 dicts, each with keys 'x', 'y', 'z' (MediaPipe
               NormalizedLandmark values forwarded by the Angular frontend).
    """
    if len(landmarks) != 21:
        return None
    wrist = landmarks[0]
    coords: list[float] = []
    for lm in landmarks:
        coords.extend([lm["x"] - wrist["x"], lm["y"] - wrist["y"], lm["z"] - wrist["z"]])
    arr = np.array(coords, dtype=np.float32)
    mid_base = landmarks[9]
    scale = float(np.sqrt((mid_base["x"] - wrist["x"]) ** 2 + (mid_base["y"] - wrist["y"]) ** 2))
    if scale > 1e-6:
        arr /= scale
    geo = _compute_geometric_features(arr.reshape(21, 3))
    return np.concatenate([arr, geo])  # 63 + 19 = 82


# ═══════════════════════════════════════════════════════════════════════════════
# SVM INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════


def _run_svm(feature_vector: np.ndarray) -> dict:
    """Run SVM pipeline on a 63-D vector → {label, confidence}.

    Returns {"label": "uncertain", ...} when confidence < CONFIDENCE_THRESHOLD
    so the backend never forwards ambiguous predictions to the frontend.
    """
    proba = _pipeline.predict_proba(feature_vector.reshape(1, -1))[0]  # type: ignore[union-attr]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx]) * 100
    if confidence < CONFIDENCE_THRESHOLD:
        return {"label": "uncertain", "confidence": round(confidence, 2)}
    label = _class_names[pred_idx] if pred_idx < len(_class_names) else "error"
    return {"label": label, "confidence": round(confidence, 2)}


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC PREDICT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def predict_sign(image_bytes: bytes) -> dict:
    """Predict from raw image bytes (HTTP endpoint).

    Runs MediaPipe server-side to extract landmarks, then SVM.
    Accepts any real webcam frame — no skeleton preprocessing required.

    Returns:
        {"label": str, "confidence": float (0-100)}
    """
    global _model_loaded, _pipeline

    if not _model_loaded or _pipeline is None:
        if not load_ml_model():
            return {"label": "error", "confidence": 0.0}

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"label": "error", "confidence": 0.0}

        detector = _get_detector()
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_img)

        if not result.hand_landmarks:
            return {"label": "no_hand", "confidence": 0.0}

        features = _normalize_mediapipe_result(result.hand_landmarks[0])
        return _run_svm(features)

    except Exception:
        logger.exception("predict_sign error")
        return {"label": "error", "confidence": 0.0}


def predict_sign_from_landmarks(landmarks: list[dict]) -> dict:
    """Predict from pre-extracted MediaPipe landmarks (WebSocket fast path).

    The Angular frontend already runs MediaPipe Tasks Vision for the hand
    overlay display.  Instead of re-encoding the frame as JPEG and
    re-running MediaPipe on the backend, the frontend sends the 21
    NormalizedLandmark dicts directly — ~500 B JSON vs ~30 KB JPEG.

    Args:
        landmarks: list of 21 dicts with keys 'x', 'y', 'z'

    Returns:
        {"label": str, "confidence": float (0-100)}
    """
    global _model_loaded, _pipeline

    if not _model_loaded or _pipeline is None:
        if not load_ml_model():
            return {"label": "error", "confidence": 0.0}

    try:
        features = _normalize_landmarks_json(landmarks)
        if features is None:
            return {"label": "no_hand", "confidence": 0.0}
        return _run_svm(features)

    except Exception:
        logger.exception("predict_sign_from_landmarks error")
        return {"label": "error", "confidence": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# INFO & UTILITY  (same interface as before — no API changes needed)
# ═══════════════════════════════════════════════════════════════════════════════


def get_public_info() -> dict:
    return {
        "available": _model_loaded,
        "supported_gestures": _class_names,
        "total_gestures": len(_class_names),
    }


def get_health_status() -> dict:
    return {
        "status": "healthy" if _model_loaded else "unhealthy",
        "service": "gesture-prediction",
    }


def reload_model() -> dict:
    global _pipeline, _class_names, _model_loaded
    _pipeline = None
    _class_names = []
    _model_loaded = False
    success = load_ml_model()
    return {
        "success": success,
        "message": "Model reloaded successfully" if success else "Failed to reload model",
    }
