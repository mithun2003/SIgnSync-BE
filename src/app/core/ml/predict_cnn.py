import logging

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from .schema import PredictData

logger = logging.getLogger(__name__)

# MODEL_PATH = "./trained_model/sign_language_mobilenet.h5"
# CLASS_PATH = "./trained_model/class_names.txt"
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "trained_model" / "sign_language_mobilenet.keras"
CLASS_PATH = BASE_DIR / "trained_model" / "class_names.txt"
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH) as f:
    CLASS_NAMES = [line.strip() for line in f]

IMG_SIZE = (224, 224)


def predict_image(image_bytes: bytes) -> PredictData:
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    idx = np.argmax(preds)
    confidence = float(preds[0][idx])
    logger.debug(
        "Prediction: idx=%d label=%s confidence=%.4f (%.2f%%)",
        idx,
        CLASS_NAMES[idx],
        confidence,
        confidence * 100,
    )

    return PredictData(label=CLASS_NAMES[idx], confidence=round(confidence * 100, 2))
