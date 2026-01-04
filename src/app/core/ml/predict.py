import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from .schema import PredictData

MODEL_PATH = "/trained_model/sign_language_mobilenet.h5"
CLASS_PATH = "/trained_model/class_names.txt"

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

    return {"label": CLASS_NAMES[idx], "confidence": round(confidence, 4)}
