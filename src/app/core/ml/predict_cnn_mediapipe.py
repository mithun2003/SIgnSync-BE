import logging

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model" / "sign_language_mobilenet.keras"
CLASS_PATH = BASE_DIR / "trained_model" / "class_names.txt"
TASK_PATH = BASE_DIR / "trained_model" / "hand_landmarker.task"

IMG_SIZE = (224, 224)

# --- 2. LOAD MODELS ---
# Load CNN (The Sign Recognizer)
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("CNN model loaded successfully")
except Exception as e:
    logger.exception("CNN model load error: %s", e)
    model = None

# Load Class Names
try:
    with open(CLASS_PATH) as f:
        CLASS_NAMES = [line.strip() for line in f]
except Exception as e:
    logger.exception("Class name load error: %s", e)
    CLASS_NAMES = []

# Load MediaPipe (The Hand Cropper)
try:
    base_options = python.BaseOptions(model_asset_path=str(TASK_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,  # Low threshold to catch difficult angles
    )
    detector = vision.HandLandmarker.create_from_options(options)
    logger.info("MediaPipe hand landmarker loaded successfully")
except Exception as e:
    logger.exception("MediaPipe load error: %s", e)
    detector = None


# --- 3. HELPER: CROP HAND ---
def get_cropped_image(img):
    """Accepts a full image (cv2 BGR).

    Returns:
       - The cropped hand (cv2 BGR) if found.
       - None if no hand is found.
    """
    if detector is None:
        return None

    # MediaPipe expects RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Detect
    detection_result = detector.detect(mp_image)

    if not detection_result.hand_landmarks:
        return None  # No hand found

    # Calculate Box
    landmarks = detection_result.hand_landmarks[0]
    h, w, _ = img.shape
    x_vals = [lm.x for lm in landmarks]
    y_vals = [lm.y for lm in landmarks]

    min_x, max_x = int(min(x_vals) * w), int(max(x_vals) * w)
    min_y, max_y = int(min(y_vals) * h), int(max(y_vals) * h)

    # Add Padding (Important!)
    padding = 100
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(w, max_x + padding)
    max_y = min(h, max_y + padding)

    # Crop
    cropped = img[min_y:max_y, min_x:max_x]

    if cropped.size == 0:
        return None

    return cropped


# --- 4. PREDICTION FUNCTION ---
DEBUG_MODE = True


def predict_image(image_bytes: bytes):
    if model is None:
        return {"label": "Error", "confidence": 0.0}

    # A. Decode
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # B. FIX MIRRORING (Crucial for Webcams!)
    # Flip the image horizontally to match real life
    img = cv2.flip(img, 1)

    # C. CROP HAND
    cropped_img = get_cropped_image(img)

    if cropped_img is not None:
        final_img = cropped_img

        # --- DEBUGGING: SAVE THE CROP ---
        # If this image looks weird (fingers cut off), increase 'padding' in get_cropped_image
        if DEBUG_MODE:
            cv2.imwrite("debug_crop.jpg", final_img)
            logger.debug("Debug crop saved to debug_crop.jpg")

    else:
        return {"label": "No Hand Detected", "confidence": 0.0}

    # D. Preprocess
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    final_img = cv2.resize(final_img, IMG_SIZE)
    final_img = final_img.astype(np.float32)
    final_img = preprocess_input(final_img)
    final_img = np.expand_dims(final_img, axis=0)

    # E. Predict
    preds = model.predict(final_img, verbose=0)
    idx = np.argmax(preds)
    confidence = float(preds[0][idx])

    return {"label": CLASS_NAMES[idx], "confidence": round(confidence * 100, 2)}
