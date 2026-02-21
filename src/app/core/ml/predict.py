"""Sign Language Gesture Prediction Module Handles model loading and prediction for skeleton images."""

import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "trained_model"
MODEL_PATH = MODELS_DIR / "sign_language_mobilenet.keras"
CLASS_PATH = MODELS_DIR / "class_names.json"

# Image settings
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 30.0  # Minimum confidence percentage (0-100)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

print(f"⏳ Looking for models in: {MODELS_DIR}")

model: object | None = None
class_names: list[str] = []
model_loaded: bool = False


def load_ml_model():
    """Load the ML model and class names."""
    global model, class_names, model_loaded

    if not MODEL_PATH.exists():
        print(f"❌ CRITICAL: Model not found at {MODEL_PATH}")
        return False

    if not CLASS_PATH.exists():
        print(f"❌ CRITICAL: Class names not found at {CLASS_PATH}")
        return False

    try:
        # Load model
        model = load_model(str(MODEL_PATH))
        print(f"✅ Model loaded from: {MODEL_PATH}")

        # Load class names from JSON
        with open(CLASS_PATH) as f:
            class_names = json.load(f)

        print(f"✅ Classes loaded: {len(class_names)} → {class_names}")
        model_loaded = True
        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_loaded = False
        return False


# Load on module import
# load_ml_model()

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
    mean_value = np.mean(img)

    if mean_value < 1:
        return False  # Completely black = no hand detected

    return True


def preprocess_skeleton_image(img: np.ndarray) -> np.ndarray:
    """Preprocess skeleton image for prediction.

    - Convert to RGB if needed
    - Resize to model input size (224x224)
    - Normalize to [0, 1]
    - Add batch dimension
    """
    # Ensure RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img_resized = cv2.resize(img, IMG_SIZE)

    # Normalize to [0, 1]
    img_array = img_to_array(img_resized) / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch


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

    # Check if model is loaded
    if not model_loaded or model is None:
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
        img_batch = preprocess_skeleton_image(img)

        # 4. Run prediction
        predictions = model.predict(img_batch, verbose=0)[0]

        # 5. Get top prediction
        pred_idx = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100
        label = class_names[pred_idx]

        # 6. Check confidence threshold
        # if confidence < CONFIDENCE_THRESHOLD:
        #     return {"label": "uncertain", "confidence": round(confidence, 2)}

        # 7. Return clean result
        return {"label": label, "confidence": round(confidence, 2)}

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {"label": "error", "confidence": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# INFO & UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def get_public_info() -> dict:
    """Get PUBLIC information about the prediction service."""
    global class_names, model_loaded

    return {"available": model_loaded, "supported_gestures": class_names, "total_gestures": len(class_names)}


def get_health_status() -> dict:
    """Simple health check."""
    global model_loaded

    return {"status": "healthy" if model_loaded else "unhealthy", "service": "gesture-prediction"}


def reload_model() -> dict:
    """Reload the model (for hot-reloading)"""
    success = load_ml_model()
    return {"success": success, "message": "Model reloaded successfully" if success else "Failed to reload model"}
