import os
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent


# Point to the 'trained_models' folder
MODELS_DIR = os.path.join(BASE_DIR, "trained_model")
MODEL_PATH = os.path.join(MODELS_DIR, "sign_language_mobilenet.keras")
CLASS_PATH = os.path.join(MODELS_DIR, "class_names.txt")

# --- GLOBAL LOADERS ---
print(f"⏳ Looking for models in: {MODELS_DIR}")

model = None
class_names = []

if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_PATH):
    try:
        model = load_model(MODEL_PATH)
        with open(CLASS_PATH) as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"✅ Model Loaded! Classes: {len(class_names)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ CRITICAL: Files not found at {MODEL_PATH}")


# --- PREDICTION FUNCTION ---
def predict_sign(image_bytes):
    """Expects an ALREADY CROPPED hand image from the frontend.

    Resizes, normalizes, and predicts.
    """
    if model is None:
        return {"error": "Model not loaded properly"}

    try:
        # 1. Decode Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image data"}

        # 2. Preprocess (Resize to 224x224 & Normalize)
        # Note: We do NOT crop here because you did it in Frontend
        img_resized = cv2.resize(img, (224, 224))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 3. Predict
        predictions = model.predict(img_array, verbose=0)

        # 4. Process Results
        pred_idx = np.argmax(predictions)
        confidence = float(np.max(predictions) * 100)
        label = class_names[pred_idx]

        return {"label": label, "confidence": round(confidence, 2)}

    except Exception as e:
        return {"error": str(e)}
