from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model" / "svm_landmark_model.pkl"
ENCODER_PATH = BASE_DIR / "trained_model" / "label_encoder.pkl"
TASK_PATH = BASE_DIR / "trained_model" / "hand_landmarker.task"

# --- 1. LOAD MODELS ---
try:
    svm_model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("✅ SVM Model & Encoder Loaded")
except Exception as e:
    print(f"❌ SVM Load Error: {e}")
    svm_model = None
    label_encoder = None

# --- 2. LOAD MEDIAPIPE ---
try:
    base_options = python.BaseOptions(model_asset_path=str(TASK_PATH))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.5)
    detector = vision.HandLandmarker.create_from_options(options)
    print("✅ MediaPipe Landmarker Loaded")
except Exception as e:
    print(f"❌ MediaPipe Error: {e}")
    detector = None


# --- 3. PREDICTION FUNCTION ---
def predict_image(image_bytes: bytes):
    if svm_model is None or detector is None:
        return {"label": "Error", "confidence": 0.0}

    # A. Decode Image
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # B. Fix Mirroring (Important for Webcam!)
    img = cv2.flip(img, 1)

    # C. Convert to MediaPipe Image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # D. Detect Landmarks
    detection_result = detector.detect(mp_image)

    if not detection_result.hand_landmarks:
        return {"label": "No Hand Detected", "confidence": 0.0}

    # E. Extract Features (Exact same logic as training!)
    landmarks = detection_result.hand_landmarks[0]
    feature_vector = []
    for lm in landmarks:
        feature_vector.extend([lm.x, lm.y, lm.z])  # 21 * 3 = 63 features

    # F. Predict
    # Reshape to (1, 63)
    features_np = np.array([feature_vector], dtype=np.float32)

    # Get probabilities
    probs = svm_model.predict_proba(features_np)[0]
    best_idx = np.argmax(probs)
    confidence = float(probs[best_idx])

    # Decode Label (0 -> 'A')
    label_str = label_encoder.inverse_transform([best_idx])[0]

    return {"label": label_str, "confidence": round(confidence * 100, 2)}
