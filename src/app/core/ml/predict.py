"""Sign Language Gesture Prediction — MLP (ONNX) entry point.

Delegates to sign_model_mlp.predict which runs a PyTorch MLP exported
to ONNX for zero-dependency CPU inference.

Model file:  src/app/core/ml/sign_model_mlp/trained_model/sign_language_mlp.onnx
Labels file: src/app/core/ml/sign_model_mlp/trained_model/class_names_mlp.json

Two prediction paths:
  • predict_sign(image_bytes)           — HTTP endpoint: raw image → MediaPipe → MLP
  • predict_sign_from_landmarks(lms)    — WebSocket: pre-extracted landmarks → MLP
    (frontend already runs MediaPipe; sending 63 floats is 60× cheaper than a JPEG)
"""

from .sign_model_mlp.predict import (
    get_health_status,
    get_public_info,
    load_ml_model,
    predict_sign,
    predict_sign_from_landmarks,
    reload_model,
)

__all__ = [
    "load_ml_model",
    "predict_sign",
    "predict_sign_from_landmarks",
    "get_public_info",
    "get_health_status",
    "reload_model",
]
