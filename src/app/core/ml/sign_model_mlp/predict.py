"""Sign Language Gesture Prediction — MLP (ONNX runtime).

Fallback model: PyTorch MLP exported to ONNX for zero-dependency inference.
Model file:  src/app/core/ml/trained_model/sign_language_mlp.onnx
Labels file: src/app/core/ml/trained_model/class_names_mlp.json

Same public interface as src/app/core/ml/predict.py (SVM) — drop-in replacement.
Two prediction paths:
  • predict_sign(image_bytes)           — HTTP endpoint: raw image → MediaPipe → MLP
  • predict_sign_from_landmarks(lms)    — WebSocket: pre-extracted landmarks → MLP
"""

import json
import logging
import threading
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision

from .config import LABELS_NAME, MODEL_DIR, ONNX_MODEL_NAME

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = MODEL_DIR / ONNX_MODEL_NAME
CLASS_PATH = MODEL_DIR / LABELS_NAME

CONFIDENCE_THRESHOLD = 85.0

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)

_session: ort.InferenceSession | None = None
_class_names: list[str] = []
_model_loaded: bool = False
_model_load_lock = threading.Lock()
_input_name: str = "input"

# MediaPipe Tasks Vision — HandLandmarker (replaces deprecated mp.solutions.hands)
_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_TASK_MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

_detector: _mp_vision.HandLandmarker | None = None


def _ensure_task_model() -> str:
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
    """Load ONNX session + class names from disk (thread-safe, idempotent)."""
    global _session, _class_names, _model_loaded, _input_name

    if _model_loaded and _session is not None:
        return True

    with _model_load_lock:
        if _model_loaded and _session is not None:
            return True

        if not MODEL_PATH.exists():
            logger.critical("MLP ONNX model not found at %s", MODEL_PATH)
            logger.critical("Train it first:  python src/app/core/ml/sign_model_mlp/train_standalone.py")
            return False

        if not CLASS_PATH.exists():
            logger.critical("Class names not found at %s", CLASS_PATH)
            return False

        try:
            logger.info("Loading MLP ONNX model from %s …", MODEL_PATH)
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2
            _session = ort.InferenceSession(
                str(MODEL_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            _input_name = _session.get_inputs()[0].name
            with open(CLASS_PATH) as f:
                _class_names = json.load(f)
            _model_loaded = True
            logger.info("MLP ONNX model loaded — %d classes", len(_class_names))
            return True
        except Exception:
            logger.exception("Failed to load MLP ONNX model")
            _model_loaded = False
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# LANDMARK NORMALIZATION  (same logic as SVM predict.py)
# ═══════════════════════════════════════════════════════════════════════════════


def _normalize_mediapipe_result(lms) -> np.ndarray:
    wrist = lms[0]
    coords: list[float] = []
    for lm in lms:
        coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    arr = np.array(coords, dtype=np.float32)
    mid_base = lms[9]
    scale = float(np.sqrt((mid_base.x - wrist.x) ** 2 + (mid_base.y - wrist.y) ** 2))
    if scale > 1e-6:
        arr /= scale
    return arr


def _normalize_landmarks_json(landmarks: list[dict]) -> np.ndarray | None:
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
    return arr


# ═══════════════════════════════════════════════════════════════════════════════
# MLP INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════


def _run_mlp(feature_vector: np.ndarray) -> dict:
    """Run ONNX MLP on a (63,) float32 vector → {label, confidence}.

    Returns {"label": "uncertain", ...} when confidence < CONFIDENCE_THRESHOLD
    so the backend never forwards ambiguous predictions to the frontend.
    """
    x = feature_vector.reshape(1, -1)
    logits = _session.run(None, {_input_name: x})[0][0]  # type: ignore[union-attr]
    # Softmax from logits
    exp = np.exp(logits - logits.max())
    proba = exp / exp.sum()
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx]) * 100
    label = _class_names[pred_idx] if pred_idx < len(_class_names) else "error"
    if confidence < CONFIDENCE_THRESHOLD:
        return {"label": "uncertain", "confidence": round(confidence, 2)}
    return {"label": label, "confidence": round(confidence, 2)}


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC PREDICT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def predict_sign(image_bytes: bytes) -> dict:
    """Predict from raw image bytes (HTTP endpoint).

    Returns:
        {"label": str, "confidence": float (0-100)}
    """
    if not _model_loaded or _session is None:
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
        return _run_mlp(features)

    except Exception:
        logger.exception("predict_sign error")
        return {"label": "error", "confidence": 0.0}


def predict_sign_from_landmarks(landmarks: list[dict]) -> dict:
    """Predict from pre-extracted MediaPipe landmarks (WebSocket fast path).

    Args:
        landmarks: list of 21 dicts with keys 'x', 'y', 'z'

    Returns:
        {"label": str, "confidence": float (0-100)}
    """
    if not _model_loaded or _session is None:
        if not load_ml_model():
            return {"label": "error", "confidence": 0.0}

    try:
        features = _normalize_landmarks_json(landmarks)
        if features is None:
            return {"label": "no_hand", "confidence": 0.0}
        return _run_mlp(features)

    except Exception:
        logger.exception("predict_sign_from_landmarks error")
        return {"label": "error", "confidence": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# INFO & UTILITY
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
    global _session, _class_names, _model_loaded
    _session = None
    _class_names = []
    _model_loaded = False
    success = load_ml_model()
    return {
        "success": success,
        "message": "Model reloaded successfully" if success else "Failed to reload model",
    }
