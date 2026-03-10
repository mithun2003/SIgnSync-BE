"""
Sign Language Inference Module  (SVM primary / CNN fallback)
=============================================================

Drop-in replacement for the existing TensorFlow predict.py:

    from predict import predict_sign, load_ml_model, get_health_status, get_public_info

The SVM model works on RAW camera frames — MediaPipe extracts 63-dimensional
landmark features server-side, so the frontend can send unprocessed images.
No skeleton pre-processing needed.

API compatibility
-----------------
    load_ml_model()          → bool
    predict_sign(bytes)      → {"label": str, "confidence": float}   # 0-100
    get_health_status()      → {"status": str, "service": str}
    get_public_info()        → {"available": bool, "supported_gestures": list, "total_gestures": int}

CLI demo
--------
    python predict.py --webcam
    python predict.py --image path/to/hand.jpg
    python predict.py --webcam --model cnn
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).parent
logger = logging.getLogger(__name__)

# ── Model file paths ──────────────────────────────────────────────────────────
_SVM_PATH = _HERE / "trained_model" / "sign_language_svm.joblib"
_CNN_PATH = _HERE / "trained_model" / "sign_language_cnn.pth"
_CN_SVM = _HERE / "trained_model" / "class_names_svm.json"
_CN_CNN = _HERE / "trained_model" / "class_names.json"

# ── Shared state ──────────────────────────────────────────────────────────────
_lock = threading.Lock()
_loaded = False
_model_type = "none"  # "svm" | "cnn" | "none"
_class_names: list[str] = []

# SVM
_svm_pipeline = None

# CNN
_cnn_model = None
_cnn_transform = None
_cnn_device = None


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_svm() -> bool:
    global _svm_pipeline, _class_names, _loaded, _model_type
    try:
        import joblib

        _svm_pipeline = joblib.load(_SVM_PATH)
        _class_names = json.loads(_CN_SVM.read_text()) if _CN_SVM.exists() else []
        _model_type = "svm"
        _loaded = True
        logger.info("SVM loaded — %d classes", len(_class_names))
        return True
    except Exception as exc:
        logger.warning("SVM load failed: %s", exc)
        return False


def _load_cnn() -> bool:
    global _cnn_model, _cnn_transform, _cnn_device, _class_names, _loaded, _model_type
    try:
        import sys

        sys.path.insert(0, str(_HERE))
        import torch
        from model import SignCNN
        from torchvision import transforms

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(_CNN_PATH, map_location=device, weights_only=True)

        model = SignCNN(num_classes=ckpt["num_classes"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        img_size = ckpt.get("image_size", 224)
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        cn = ckpt.get("class_names") or (json.loads(_CN_CNN.read_text()) if _CN_CNN.exists() else [])
        _cnn_model, _cnn_transform, _cnn_device = model, transform, device
        _class_names = cn
        _model_type = "cnn"
        _loaded = True
        logger.info("CNN loaded — %d classes, device=%s", ckpt["num_classes"], device)
        return True
    except Exception as exc:
        logger.warning("CNN load failed: %s", exc)
        return False


def load_ml_model(prefer: str = "svm") -> bool:
    """Load the best available model (thread-safe, idempotent).

    Tries SVM first (default), falls back to CNN.
    Pass ``prefer="cnn"`` to reverse the order.
    """
    global _loaded
    if _loaded:
        return True
    with _lock:
        if _loaded:
            return True
        if prefer == "cnn":
            return _load_cnn() or _load_svm()
        return _load_svm() or _load_cnn()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────


def predict_sign(image_bytes: bytes) -> dict:
    """Predict the sign from a RAW camera frame.

    Accepts raw JPEG/PNG bytes (unprocessed webcam image).  MediaPipe runs
    server-side to extract hand landmarks; no skeleton pre-processing needed
    on the client.

    Args:
        image_bytes: Raw JPEG / PNG bytes.

    Returns:
        ``{"label": str, "confidence": float}``  — confidence is 0–100.
        Special labels: ``"no_hand"`` (nothing detected), ``"error"`` (failure).
    """
    if not _loaded:
        load_ml_model()
    if not _loaded:
        return {"label": "error", "confidence": 0.0}

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {"label": "error", "confidence": 0.0}

        if _model_type == "svm":
            return _predict_svm(img_bgr)
        if _model_type == "cnn":
            return _predict_cnn(img_bgr)
        return {"label": "error", "confidence": 0.0}

    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        return {"label": "error", "confidence": 0.0}


def _predict_svm(img_bgr: np.ndarray) -> dict:
    import sys

    sys.path.insert(0, str(_HERE))
    from utils import extract_landmarks_array

    vec = extract_landmarks_array(img_bgr)
    if vec is None:
        return {"label": "no_hand", "confidence": 0.0}

    proba = _svm_pipeline.predict_proba(vec.reshape(1, -1))[0]
    idx = int(np.argmax(proba))
    label = _class_names[idx] if idx < len(_class_names) else str(idx)
    conf = round(float(proba[idx]) * 100.0, 2)
    return {"label": label, "confidence": conf}


def _predict_cnn(img_bgr: np.ndarray) -> dict:
    import sys

    sys.path.insert(0, str(_HERE))
    import torch
    import torch.nn.functional as F
    from utils import extract_skeleton

    skeleton = extract_skeleton(img_bgr)
    if skeleton is None:
        return {"label": "no_hand", "confidence": 0.0}

    img_rgb = cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB)
    tensor = _cnn_transform(img_rgb).unsqueeze(0).to(_cnn_device)

    with torch.no_grad():
        probs = F.softmax(_cnn_model(tensor), dim=1)[0]

    idx = probs.argmax().item()
    label = _class_names[idx] if idx < len(_class_names) else str(idx)
    conf = round(float(probs[idx]) * 100.0, 2)
    return {"label": label, "confidence": conf}


# ─────────────────────────────────────────────────────────────────────────────
# API helper functions  (required by existing API router)
# ─────────────────────────────────────────────────────────────────────────────


def get_public_info() -> dict:
    """Return service info for the /predict/info endpoint."""
    return {
        "available": _loaded,
        "supported_gestures": _class_names,
        "total_gestures": len(_class_names),
    }


def get_health_status() -> dict:
    """Return health check dict for the /predict/health endpoint."""
    return {
        "status": "healthy" if _loaded else "unhealthy",
        "service": "gesture-prediction",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────


def _demo_webcam() -> None:
    import sys

    sys.path.insert(0, str(_HERE))
    from utils import extract_skeleton

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    print("Live demo — press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        _, buf = cv2.imencode(".jpg", frame)
        result = predict_sign(buf.tobytes())
        label, conf = result["label"], result["confidence"]

        color = (0, 220, 0) if conf > 75 else (0, 165, 255)
        cv2.putText(display, f"{label}  {conf:.1f}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

        # Show skeleton thumbnail if SVM found a hand
        if label not in ("no_hand", "error"):
            skeleton = extract_skeleton(frame)
            if skeleton is not None:
                thumb = cv2.resize(skeleton, (150, 150))
                display[10:160, -160:-10] = thumb

        cv2.imshow("SignSync — live", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def _demo_image(path: str) -> None:
    import sys

    sys.path.insert(0, str(_HERE))
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot read: {path}")
        return
    _, buf = cv2.imencode(".jpg", img)
    result = predict_sign(buf.tobytes())
    print(f"Prediction: {result['label']}  ({result['confidence']:.1f}%)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--model", choices=["svm", "cnn"], default="svm")
    args = ap.parse_args()

    load_ml_model(prefer=args.model)

    if args.image:
        _demo_image(args.image)
    elif args.webcam:
        _demo_webcam()
    else:
        print("Use --webcam or --image <path>")
