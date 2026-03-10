"""Sign Language Gesture Prediction Module Handles model loading and prediction for skeleton images."""

import json
import logging
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B3_Weights

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "trained_model"
MODEL_PATH = MODELS_DIR / "sign_language_mobilenet.pth"
CLASS_PATH = MODELS_DIR / "class_names.json"

# Image settings
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 30.0  # Minimum confidence percentage (0-100)

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE & TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL STATE
# ═══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)
logger.debug("Looking for models in: %s", MODELS_DIR)

model: nn.Module | None = None
class_names: list[str] = []
model_loaded: bool = False
_model_load_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE  (must match train_v2_pytorch.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_model(num_classes: int) -> nn.Module:
    """Recreate EfficientNetB3 + custom head — identical to training script."""
    net = models.efficientnet_b3(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return net


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_ml_model() -> bool:
    global model, class_names, model_loaded

    if model_loaded and model is not None:
        return True

    with _model_load_lock:
        # Double-checked locking: re-verify after acquiring the lock
        if model_loaded and model is not None:
            return True

        if not MODEL_PATH.exists():
            logger.critical("Model not found at %s", MODEL_PATH)
            return False

        if not CLASS_PATH.exists():
            logger.critical("Class names not found at %s", CLASS_PATH)
            return False

        try:
            logger.info("Loading ML model...")

            with open(CLASS_PATH) as f:
                class_names = json.load(f)

            # Build architecture then load saved weights
            net = _build_model(len(class_names))
            state_dict = torch.load(str(MODEL_PATH), map_location=DEVICE)
            net.load_state_dict(state_dict)
            net.to(DEVICE)
            net.eval()  # disables dropout / batchnorm training mode

            model = net
            model_loaded = True
            logger.info("ML model loaded on %s (%d classes)", DEVICE, len(class_names))
            return True

        except Exception as e:
            logger.exception("Error loading ML model: %s", e)
            model_loaded = False
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_skeleton_image(img: np.ndarray) -> bool:
    """Validate if the image is a valid skeleton image.

    Skeleton images should have white lines on black background.
    """
    if img is None:
        return False

    # Check if image is mostly black (skeleton on black background)
    # But not completely black (mean > 1)
    if np.mean(img) < 1:
        return False  # Completely black = no hand detected

    return True


def preprocess_skeleton_image(img: np.ndarray) -> torch.Tensor:
    """Preprocess skeleton image for prediction.

    - Convert BGR → RGB
    - Resize to 224×224
    - Normalize with ImageNet mean/std
    - Add batch dimension
    """
    # Ensure RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = _TRANSFORM(img)        # (3, 224, 224)
    return tensor.unsqueeze(0)      # (1, 3, 224, 224)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict_sign(image_bytes: bytes) -> dict:
    """Predict sign language gesture from skeleton image bytes.

    Args:
        image_bytes: Raw image bytes (skeleton image from frontend)

    Returns:
        dict with keys:
            - label: Predicted gesture (e.g., 'A', 'B', 'hello')
            - confidence: Confidence percentage (0-100), rounded to 2 decimals
    """
    global model, class_names, model_loaded

    # Lazy load model if needed
    if not model_loaded or model is None:
        if not load_ml_model():
            return {"label": "error", "confidence": 0.0}

    try:
        # 1. Decode image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"label": "error", "confidence": 0.0}

        # 2. Validate skeleton image
        if not validate_skeleton_image(img):
            return {"label": "no_hand", "confidence": 0.0}

        # 3. Preprocess for model
        tensor = preprocess_skeleton_image(img).to(DEVICE)

        # 4. Run prediction (no gradients needed for inference)
        with torch.no_grad():
            logits = model(tensor)                      # (1, num_classes)
            probs  = torch.softmax(logits, dim=1)[0]    # (num_classes,)

        # 5. Get top prediction
        pred_idx   = int(probs.argmax())
        confidence = float(probs[pred_idx]) * 100
        label      = class_names[pred_idx]

        # 6. Check confidence threshold
        # if confidence < CONFIDENCE_THRESHOLD:
        #     return {"label": "uncertain", "confidence": round(confidence, 2)}

        # 7. Return clean result
        return {"label": label, "confidence": round(confidence, 2)}

    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return {"label": "error", "confidence": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# INFO & UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_public_info() -> dict:
    """Get PUBLIC information about the prediction service."""
    global class_names, model_loaded

    return {
        "available": model_loaded,
        "supported_gestures": class_names,
        "total_gestures": len(class_names),
        "device": str(DEVICE),
    }


def get_health_status() -> dict:
    """Simple health check."""
    global model_loaded

    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "service": "gesture-prediction",
        "device": str(DEVICE),
    }


def reload_model() -> dict:
    """Reload the model (for hot-reloading)"""
    global model, model_loaded
    model = None
    model_loaded = False
    success = load_ml_model()
    return {"success": success, "message": "Model reloaded successfully" if success else "Failed to reload model"}