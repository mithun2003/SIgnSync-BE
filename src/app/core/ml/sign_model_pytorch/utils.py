"""
Utility functions: MediaPipe hand skeleton extraction and landmark array extraction.
"""

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision

# Hand bone connections — identical to the old mp.solutions.hands.HAND_CONNECTIONS
_HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
]

_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_TASK_MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

_detector: _mp_vision.HandLandmarker | None = None


def _ensure_task_model() -> str:
    if not _TASK_MODEL_PATH.exists():
        print("Downloading hand_landmarker.task …")
        urllib.request.urlretrieve(_HAND_LANDMARKER_URL, _TASK_MODEL_PATH)
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


def extract_skeleton(image_bgr: np.ndarray, size: int = 224) -> np.ndarray | None:
    """Convert a BGR image to a white-on-black hand skeleton image.

    Detects the hand with MediaPipe, draws connections and joint dots on a
    black canvas cropped tightly around the hand.

    Args:
        image_bgr: Input BGR image (any resolution).
        size:      Output canvas size (square, default 224).

    Returns:
        A ``size × size`` BGR skeleton image, or ``None`` if no hand detected.
    """
    detector = _get_detector()
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_img)

    if not result.hand_landmarks:
        return None

    lms = result.hand_landmarks[0]
    h, w = image_bgr.shape[:2]

    # Compute tight bounding box around the hand
    xs = [lm.x * w for lm in lms]
    ys = [lm.y * h for lm in lms]
    x_min, x_max = max(0, int(min(xs))), min(w, int(max(xs)))
    y_min, y_max = max(0, int(min(ys))), min(h, int(max(ys)))

    # Add 15 % padding
    pad_x = int((x_max - x_min) * 0.15) + 5
    pad_y = int((y_max - y_min) * 0.15) + 5
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)

    box_w = max(x_max - x_min, 1)
    box_h = max(y_max - y_min, 1)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    def _to_canvas(lm) -> tuple[int, int]:
        cx = int((lm.x * w - x_min) / box_w * (size - 1))
        cy = int((lm.y * h - y_min) / box_h * (size - 1))
        return (int(np.clip(cx, 0, size - 1)), int(np.clip(cy, 0, size - 1)))

    # Draw bone connections
    for start_idx, end_idx in _HAND_CONNECTIONS:
        p1 = _to_canvas(lms[start_idx])
        p2 = _to_canvas(lms[end_idx])
        cv2.line(canvas, p1, p2, (255, 255, 255), 2)

    # Draw joints
    for lm in lms:
        cv2.circle(canvas, _to_canvas(lm), 3, (180, 180, 180), -1)

    return canvas


def extract_landmarks_array(image_bgr: np.ndarray) -> np.ndarray | None:
    """Extract normalised hand landmarks as a flat feature vector for SVM.

    Computes 21 landmark positions relative to the wrist, then scales by the
    wrist-to-middle-finger-base distance so the result is scale-invariant.

    Args:
        image_bgr: Input BGR image.

    Returns:
        Float32 array of shape ``(63,)`` (21 landmarks × 3 coords),
        or ``None`` if no hand detected.
    """
    detector = _get_detector()
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_img)

    if not result.hand_landmarks:
        return None

    lms = result.hand_landmarks[0]
    wrist = lms[0]

    coords: list[float] = []
    for lm in lms:
        coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

    arr = np.array(coords, dtype=np.float32)

    # Scale by wrist→middle-finger-base distance for size invariance
    mid_base = lms[9]
    scale = float(np.sqrt((mid_base.x - wrist.x) ** 2 + (mid_base.y - wrist.y) ** 2))
    if scale > 1e-6:
        arr /= scale

    return arr
