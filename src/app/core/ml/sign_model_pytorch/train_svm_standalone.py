#!/usr/bin/env python3
"""
ASL Sign Language SVM Trainer — Standalone / Colab-ready
=========================================================

NO GPU REQUIRED.  SVM is a CPU algorithm.
Free Colab CPU runtime trains this in ~5-10 minutes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GOOGLE COLAB — Quick Start
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 — Install dependencies (run in a Colab cell):

    !pip install mediapipe scikit-learn opencv-python-headless tqdm kagglehub joblib

Step 2 — Upload your Kaggle API key (one-time):

    from google.colab import files
    files.upload()                          # upload kaggle.json
    !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

Step 3 — Upload this file and run:

    !python train_svm_standalone.py                        # auto-downloads dataset
    !python train_svm_standalone.py --data-dir /my/path   # use existing dataset
    !python train_svm_standalone.py --test                 # smoke-test (3 classes)

Step 4 — Download the trained model:

    from google.colab import files
    files.download('trained_model/sign_language_svm.joblib')
    files.download('trained_model/class_names_svm.json')

Step 5 — On your target device (NO dataset, NO GPU needed):

    pip install mediapipe scikit-learn opencv-python joblib
    # copy sign_language_svm.joblib + class_names_svm.json
    # run your predict script

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
33 CLASSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  A-Z (26)       — real images from Kaggle ASL dataset
  space, del (2) — real images from Kaggle ASL dataset
  help           — real images: "five" gesture (open palm, all fingers spread)
  danger         — synthetic fallback (ilv gesture is not available in the dataset)
  emergency      — real images: "zero" gesture (fist)
  thumbs_down    — real images: "down" gesture (not ok)
  ok_sign        — real images: "up" gesture (thumbs up)

Why SVM beats CNN for hand signs:
  CNN (64×64×3 = 12,288 inputs): sees background, lighting, clothes
  SVM (21 landmarks × 3 = 63 inputs): sees ONLY joint angles — pure signal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Config (inlined — no separate config.py needed)
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
KAGGLE_DATASET = "grassknoted/asl-alphabet"
MODEL_DIR = Path("trained_model")
EMERGENCY_DATASET = "anoshal/hand-gesture-recognition-dataset-one-hand"

ASL_CLASSES: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del"]

EMERGENCY_CLASSES: list[str] = [
    "help",
    "danger",
    "emergency",
    "thumbs_down",
    "ok_sign",
]

GESTURE_TO_EMERGENCY_MAP: dict[str, str] = {
    "down": "thumbs_down",
    "five": "help",
    "up": "ok_sign",
    "zero": "emergency",
}

ALL_CLASSES: list[str] = ASL_CLASSES + EMERGENCY_CLASSES  # 33 total


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe landmark extractor (inlined — no separate utils.py needed)
# ─────────────────────────────────────────────────────────────────────────────
#
# Normalisation applied to every real AND synthetic sample:
#   1. Subtract wrist (landmark 0)  →  wrist always at (0, 0, 0)
#   2. Divide by wrist-to-middle-MCP distance  →  scale invariant
#   3. y is NEGATIVE upward (image origin = top-left, y grows downward)

_mp_tasks = __import__("mediapipe.tasks", fromlist=["python"]).python
_mp_vision = __import__("mediapipe.tasks.python", fromlist=["vision"]).vision

_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
try:
    _TASK_MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
except NameError:
    # Running inside Colab / Jupyter where __file__ is not defined
    _TASK_MODEL_PATH = Path.cwd() / "hand_landmarker.task"
_detector: object | None = None


def _ensure_task_model() -> str:
    if not _TASK_MODEL_PATH.exists():
        import urllib.request

        print("Downloading hand_landmarker.task …")
        urllib.request.urlretrieve(_HAND_LANDMARKER_URL, _TASK_MODEL_PATH)
    return str(_TASK_MODEL_PATH)


def _get_detector():
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


def extract_landmarks_array(image_bgr: np.ndarray) -> np.ndarray | None:
    """Return normalised (82,) float32 feature vector, or None if no hand.

    Feature layout (must match predict.py exactly):   [0:63]  — 21 landmarks × (x,y,z), wrist-subtracted + scale-
    normalised   [63:73] — 10 pairwise tip-to-tip distances   [73:78] — 5  finger extension scores   [78:82] — 4
    adjacent tip lateral spreads
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
    mid_base = lms[9]
    scale = float(np.sqrt((mid_base.x - wrist.x) ** 2 + (mid_base.y - wrist.y) ** 2))
    if scale > 1e-6:
        arr /= scale
    geo = _compute_geometric_features(arr.reshape(21, 3))
    return np.concatenate([arr, geo])  # 82-D


# ─────────────────────────────────────────────────────────────────────────────
# Geometric feature helper  (shared by training + inference)
# ─────────────────────────────────────────────────────────────────────────────
# Landmark indices
_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
_MCPS = [2, 5, 9, 13, 17]  # thumb CMC + finger MCPs
_TIP_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]  # 10 pairs


def _compute_geometric_features(pts_norm: np.ndarray) -> np.ndarray:
    """Compute 19 geometric features from a (21, 3) normalised landmark array.

    Makes visually similar signs (e.g. B vs W) easily separable:
    - Tip distances:   B has tightly clustered tips; W has spread-out tips
    - Extension scores: distinguish curled vs straight fingers
    - Lateral spreads:  directly encode horizontal finger spread pattern
    """
    tip_pts = pts_norm[_TIPS]
    mcp_pts = pts_norm[_MCPS]
    features: list[float] = []

    for i, j in _TIP_PAIRS:  # 10 pairwise distances
        diff = tip_pts[i] - tip_pts[j]
        features.append(float(np.sqrt(np.dot(diff, diff))))

    for i in range(5):  # 5 extension scores
        features.append(float(mcp_pts[i, 1] - tip_pts[i, 1]))

    for i in range(4):  # 4 lateral spreads
        features.append(float(tip_pts[i, 0] - tip_pts[i + 1, 0]))

    return np.array(features, dtype=np.float32)  # 19 values


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic emergency sign landmark definitions
# ─────────────────────────────────────────────────────────────────────────────
#
# Coordinates are in the SAME normalised space as extract_landmarks_array:
#   • Wrist subtracted → wrist at (0, 0, 0)
#   • Divided by wrist-to-middle-MCP distance
#   • y is NEGATIVE upward
#
# MediaPipe landmark order (21 × 3 = 63 values):
#   0: WRIST
#   1-4:  THUMB  (CMC, MCP, IP, TIP)
#   5-8:  INDEX  (MCP, PIP, DIP, TIP)
#   9-12: MIDDLE (MCP, PIP, DIP, TIP)  ← middle-MCP ≈ -1.0 y (scale ref)
#   13-16:RING   (MCP, PIP, DIP, TIP)
#   17-20:PINKY  (MCP, PIP, DIP, TIP)

_SYNTHETIC_SIGNS: dict[str, np.ndarray] = {
    # ── HELP: Open flat palm, all 5 fingers fully spread ─────────────────────
    # Maximum finger extension — universally recognisable "stop/help" gesture.
    "help": np.array(
        [
            [0.00, 0.00, 0.000],  # wrist
            [0.42, -0.12, 0.020],  # thumb CMC
            [0.62, -0.32, -0.040],  # thumb MCP
            [0.74, -0.54, -0.090],  # thumb IP
            [0.84, -0.74, -0.110],  # thumb TIP
            [0.33, -0.93, 0.000],  # index MCP
            [0.33, -1.52, -0.070],  # index PIP
            [0.33, -1.92, -0.110],  # index DIP
            [0.33, -2.23, -0.130],  # index TIP
            [0.05, -1.00, 0.000],  # middle MCP  (scale reference)
            [0.05, -1.65, -0.070],  # middle PIP
            [0.05, -2.05, -0.110],  # middle DIP
            [0.05, -2.37, -0.130],  # middle TIP
            [-0.21, -0.95, 0.000],  # ring MCP
            [-0.21, -1.58, -0.070],  # ring PIP
            [-0.21, -1.97, -0.110],  # ring DIP
            [-0.21, -2.27, -0.130],  # ring TIP
            [-0.48, -0.84, 0.000],  # pinky MCP
            [-0.48, -1.36, -0.070],  # pinky PIP
            [-0.48, -1.64, -0.110],  # pinky DIP
            [-0.48, -1.88, -0.130],  # pinky TIP
        ],
        dtype=np.float32,
    ),
    # ── DANGER: ILY sign — index + pinky + thumb out, middle + ring curled ───
    # "I Love You" repurposed as distress. Not in any A-Z ASL shape.
    "danger": np.array(
        [
            [0.00, 0.00, 0.000],  # wrist
            [0.42, -0.12, 0.020],  # thumb CMC
            [0.62, -0.32, -0.040],  # thumb MCP  (extended outward)
            [0.74, -0.54, -0.090],  # thumb IP
            [0.84, -0.74, -0.110],  # thumb TIP
            [0.33, -0.93, 0.000],  # index MCP
            [0.33, -1.52, -0.070],  # index PIP  (extended up)
            [0.33, -1.92, -0.110],  # index DIP
            [0.33, -2.23, -0.130],  # index TIP
            [0.05, -1.00, 0.000],  # middle MCP
            [0.02, -0.79, 0.140],  # middle PIP  (curled INTO palm)
            [-0.02, -0.57, 0.190],  # middle DIP
            [-0.04, -0.42, 0.210],  # middle TIP
            [-0.21, -0.95, 0.000],  # ring MCP
            [-0.24, -0.73, 0.140],  # ring PIP    (curled)
            [-0.29, -0.52, 0.190],  # ring DIP
            [-0.31, -0.38, 0.210],  # ring TIP
            [-0.48, -0.84, 0.000],  # pinky MCP
            [-0.48, -1.36, -0.070],  # pinky PIP  (extended up)
            [-0.48, -1.64, -0.110],  # pinky DIP
            [-0.48, -1.88, -0.130],  # pinky TIP
        ],
        dtype=np.float32,
    ),
    # ── EMERGENCY: Thumbs-up fist — thumb pointing STRAIGHT up ───────────────
    # Strong, globally understood signal.
    # Key difference from ASL 'A': thumb points straight UP, not sideways.
    "emergency": np.array(
        [
            [0.00, 0.00, 0.000],  # wrist
            [0.20, -0.20, 0.000],  # thumb CMC  (going straight up)
            [0.10, -0.55, -0.050],  # thumb MCP
            [0.05, -0.95, -0.090],  # thumb IP
            [0.03, -1.35, -0.110],  # thumb TIP  (far up — NOT sideways like 'A')
            [0.33, -0.93, 0.000],  # index MCP
            [0.20, -0.74, 0.170],  # index PIP  (curled)
            [0.10, -0.54, 0.220],  # index DIP
            [0.06, -0.41, 0.230],  # index TIP
            [0.05, -1.00, 0.000],  # middle MCP
            [-0.04, -0.80, 0.170],  # middle PIP  (curled)
            [-0.11, -0.58, 0.220],  # middle DIP
            [-0.14, -0.44, 0.230],  # middle TIP
            [-0.21, -0.95, 0.000],  # ring MCP
            [-0.29, -0.74, 0.170],  # ring PIP    (curled)
            [-0.34, -0.54, 0.220],  # ring DIP
            [-0.37, -0.41, 0.230],  # ring TIP
            [-0.48, -0.84, 0.000],  # pinky MCP
            [-0.53, -0.64, 0.170],  # pinky PIP  (curled)
            [-0.56, -0.47, 0.220],  # pinky DIP
            [-0.56, -0.35, 0.230],  # pinky TIP
        ],
        dtype=np.float32,
    ),
    # ── THUMBS_DOWN: Fist with thumb pointing straight DOWN ──────────────────
    # In our coord system y POSITIVE = downward.
    # Opposite of EMERGENCY. Zero overlap with any A-Z ASL letter.
    "thumbs_down": np.array(
        [
            [0.00, 0.00, 0.000],  # wrist
            [0.20, 0.18, 0.000],  # thumb CMC  (going downward = +y)
            [0.10, 0.52, 0.050],  # thumb MCP
            [0.05, 0.90, 0.090],  # thumb IP
            [0.03, 1.32, 0.110],  # thumb TIP  (pointing DOWN = large +y)
            [0.33, -0.93, 0.000],  # index MCP
            [0.20, -0.74, 0.170],  # index PIP  (curled)
            [0.10, -0.54, 0.220],  # index DIP
            [0.06, -0.41, 0.230],  # index TIP
            [0.05, -1.00, 0.000],  # middle MCP
            [-0.04, -0.80, 0.170],  # middle PIP  (curled)
            [-0.11, -0.58, 0.220],  # middle DIP
            [-0.14, -0.44, 0.230],  # middle TIP
            [-0.21, -0.95, 0.000],  # ring MCP
            [-0.29, -0.74, 0.170],  # ring PIP    (curled)
            [-0.34, -0.54, 0.220],  # ring DIP
            [-0.37, -0.41, 0.230],  # ring TIP
            [-0.48, -0.84, 0.000],  # pinky MCP
            [-0.53, -0.64, 0.170],  # pinky PIP  (curled)
            [-0.56, -0.47, 0.220],  # pinky DIP
            [-0.56, -0.35, 0.230],  # pinky TIP
        ],
        dtype=np.float32,
    ),
    # ── OK_SIGN: Thumb+index circle, middle/ring/pinky curled DOWN ────────────
    # Critical distinction from ASL F:
    #   ASL F → middle/ring/pinky point UP (negative y tips)
    #   OK_SIGN → middle/ring/pinky curl DOWN (positive y tips)  ← very different
    "ok_sign": np.array(
        [
            [0.00, 0.00, 0.000],  # wrist
            [0.42, -0.12, 0.020],  # thumb CMC
            [0.55, -0.38, -0.030],  # thumb MCP  (curving toward index)
            [0.52, -0.58, 0.020],  # thumb IP   (bending to complete circle)
            [0.44, -0.72, 0.040],  # thumb TIP  (meeting index tip)
            [0.33, -0.93, 0.000],  # index MCP
            [0.38, -0.72, -0.030],  # index PIP  (curving toward thumb)
            [0.44, -0.60, 0.010],  # index DIP
            [0.44, -0.72, 0.040],  # index TIP  (meeting thumb tip)
            [0.05, -1.00, 0.000],  # middle MCP
            [-0.02, -0.80, 0.170],  # middle PIP  (curled DOWN — NOT up like ASL F)
            [-0.08, -0.58, 0.220],  # middle DIP
            [-0.11, -0.44, 0.230],  # middle TIP
            [-0.21, -0.95, 0.000],  # ring MCP
            [-0.28, -0.74, 0.170],  # ring PIP    (curled down)
            [-0.33, -0.54, 0.220],  # ring DIP
            [-0.36, -0.41, 0.230],  # ring TIP
            [-0.48, -0.84, 0.000],  # pinky MCP
            [-0.52, -0.64, 0.170],  # pinky PIP  (curled down)
            [-0.55, -0.47, 0.220],  # pinky DIP
            [-0.55, -0.35, 0.230],  # pinky TIP
        ],
        dtype=np.float32,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data augmentation
# ─────────────────────────────────────────────────────────────────────────────


def generate_synthetic_samples(
    base_pose: np.ndarray,
    n_samples: int = 600,
    noise_std: float = 0.055,
) -> np.ndarray:
    """Augment one canonical pose into n_samples realistic 82-D vectors.

    Augmentations applied per sample:
      1. Gaussian noise (σ=noise_std)  — MediaPipe jitter
      2. Random XY rotation ±20°       — tilted hand
      3. Random scale ±15%             — varying camera distance
      4. X-axis mirror (every other)   — left-hand users
      5. Z-axis rotation ±10°          — hand depth variation (new)

    Returns (n_samples, 82) array with geometric features appended.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    samples: list[np.ndarray] = []
    for i in range(n_samples):
        noisy = base_pose + rng.normal(0, noise_std, base_pose.shape).astype(np.float32)

        # XY rotation (hand tilt)
        angle = rng.uniform(-np.pi / 9, np.pi / 9)
        ca, sa = float(np.cos(angle)), float(np.sin(angle))
        rotated = noisy.copy()
        rotated[:, 0] = noisy[:, 0] * ca - noisy[:, 1] * sa
        rotated[:, 1] = noisy[:, 0] * sa + noisy[:, 1] * ca

        # XZ rotation (depth variation — new)
        angle_z = rng.uniform(-np.pi / 18, np.pi / 18)
        caz, saz = float(np.cos(angle_z)), float(np.sin(angle_z))
        tmp = rotated.copy()
        rotated[:, 0] = tmp[:, 0] * caz - tmp[:, 2] * saz
        rotated[:, 2] = tmp[:, 0] * saz + tmp[:, 2] * caz

        rotated *= float(rng.uniform(0.85, 1.15))
        if i % 2 == 1:
            rotated[:, 0] = -rotated[:, 0]  # mirror for left-hand users

        flat = rotated.flatten()  # 63-D coords
        geo = _compute_geometric_features(rotated)  # 19-D geometric
        samples.append(np.concatenate([flat, geo]))

    return np.array(samples, dtype=np.float32)  # (n_samples, 82)


# ─────────────────────────────────────────────────────────────────────────────
# ASL dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _find_asl_source(root: Path) -> Path:
    for candidate in [
        root / "asl_alphabet_train" / "asl_alphabet_train",
        root / "asl_alphabet_train",
        root,
    ]:
        if candidate.is_dir() and ((candidate / "A").exists() or (candidate / "a").exists()):
            return candidate
    for d in root.rglob("A"):
        if d.is_dir():
            return d.parent
    raise FileNotFoundError(f"Cannot find ASL class folders inside {root}")


def _download_asl() -> Path:
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed.  Run: pip install kagglehub")
        sys.exit(1)
    print(f"Downloading '{KAGGLE_DATASET}' from Kaggle …")
    try:
        return Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        print(f"Download failed: {exc}")
        print("Tip: upload kaggle.json to ~/.kaggle/  OR  pass --data-dir <path>")
        sys.exit(1)


def _download_emergency_dataset() -> Path | None:
    """Download hand gesture dataset for emergency signs."""
    try:
        import kagglehub
    except ImportError:
        print("Warning: kagglehub not installed. Emergency gesture images will be skipped.")
        return None

    print(f"\nAttempting to download emergency gesture dataset '{EMERGENCY_DATASET}' from Kaggle …")
    try:
        dataset_path = Path(kagglehub.dataset_download(EMERGENCY_DATASET))
        archive_path = dataset_path.parent / f"{dataset_path.name}.archive"
        if archive_path.exists():
            import tarfile

            print(f"Extracting archive: {archive_path.name}")
            try:
                with tarfile.open(archive_path, "r") as tar:
                    tar.extractall(path=dataset_path.parent)
            except Exception as exc:
                print(f"Warning: Archive extraction failed: {exc}")
        return dataset_path
    except Exception as exc:
        print(f"Warning: Emergency dataset download failed: {exc}")
        print("  → Using synthetic emergency signs instead")
        return None


def extract_emergency_landmarks(
    src_dir: Path,
    emergency_map: dict[str, str],
    emergency_classes: list[str],
    max_per_class: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract MediaPipe landmarks from gesture images for emergency classes."""
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    possible_sources = [
        src_dir / "Dataset_RGB" / "Dataset_RGB",
        src_dir / "Dataset_RGB",
        src_dir,
        src_dir / "gesture",
        src_dir / "gestures",
        src_dir / "Dataset_Binary",
    ]

    if src_dir.is_dir():
        for item in src_dir.iterdir():
            if item.is_dir():
                possible_sources.append(item)
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        possible_sources.append(subitem)

    emg_src: Path | None = None
    for cand in possible_sources:
        if not cand.is_dir():
            continue
        has_gesture = False
        for gesture_name in emergency_map:
            for variant in [gesture_name, gesture_name.upper(), gesture_name.lower()]:
                if (cand / variant).is_dir():
                    has_gesture = True
                    break
            if has_gesture:
                break
        if has_gesture:
            emg_src = cand
            print(f"  [emergency] Found gesture dataset: {cand}")
            break

    if emg_src is None:
        print(f"  [emergency] No gesture directories found in {src_dir} — will use synthetic fallback")
        return np.empty((0, 82), dtype=np.float32), np.empty(0, dtype=np.int64)

    for gesture_name, emergency_class in emergency_map.items():
        cls_idx = emergency_classes.index(emergency_class)
        gesture_dir = None
        for variant in [gesture_name, gesture_name.upper(), gesture_name.lower()]:
            cand = emg_src / variant
            if cand.is_dir():
                gesture_dir = cand
                break

        if gesture_dir is None:
            print(f"  [{emergency_class}] Gesture folder '{gesture_name}' not found — skipping")
            continue

        images = [p for p in sorted(gesture_dir.iterdir()) if p.suffix.lower() in _IMG_EXTS][:max_per_class]
        ok = fail = 0
        for img_path in tqdm(images, desc=f"  [{emergency_class}]({gesture_name})", leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                fail += 1
                continue
            vec = extract_landmarks_array(img)
            if vec is None:
                fail += 1
                continue
            X_rows.append(vec)
            y_rows.append(cls_idx)
            ok += 1

        print(f"  [{emergency_class}] {ok} landmarks from '{gesture_name}'  ({fail} skipped — no hand)")

    if not X_rows:
        return np.empty((0, 82), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int64)


def extract_asl_landmarks(
    src_dir: Path,
    classes: list[str],
    max_per_class: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract MediaPipe landmark vectors from the raw ASL image folder."""
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for cls_idx, cls in enumerate(classes):
        src_cls: Path | None = None
        for variant in [cls, cls.upper(), cls.lower()]:
            d = src_dir / variant
            if d.is_dir():
                src_cls = d
                break
        if src_cls is None:
            print(f"  [{cls}] Not found in {src_dir} — skipping.")
            continue
        images = [p for p in sorted(src_cls.iterdir()) if p.suffix.lower() in _IMG_EXTS]
        images = images[:max_per_class]
        ok = fail = 0
        for img_path in tqdm(images, desc=f"  [{cls}]", leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                fail += 1
                continue
            vec = extract_landmarks_array(img)
            if vec is None:
                fail += 1
                continue
            X_rows.append(vec)
            y_rows.append(cls_idx)
            ok += 1
        print(f"  [{cls}] {ok} landmarks extracted  ({fail} skipped — no hand detected)")
    if not X_rows:
        return np.empty((0, 82), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ASL SVM — 33 classes, CPU-only, Colab-ready",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", action="store_true", help="Quick smoke-test: only first 3 ASL classes")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip Kaggle download; search for existing dataset"
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to raw ASL dataset folder (skips download)")
    parser.add_argument(
        "--emergency-dir",
        type=Path,
        default=None,
        help="Path to emergency gesture dataset folder (skips Kaggle download for emergency signs)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1000,
        help="Max real images per ASL class  (default: 1000 — more data = better B/W separation)",
    )
    parser.add_argument(
        "--synthetic-per-sign", type=int, default=600, help="Synthetic samples per emergency sign  (default: 600)"
    )
    args, _ = parser.parse_known_args()  # ignore Jupyter/Colab kernel args (-f kernel.json)

    print("=" * 65)
    print("  ASL SVM Trainer — 33 classes | NO GPU | CPU only")
    print("=" * 65)
    print(f"  Classes       : {len(ALL_CLASSES)} total")
    print(f"  ASL (dataset) : {len(ASL_CLASSES)}")
    print(f"  Emergency      : {len(EMERGENCY_CLASSES)} (real images + synthetic fallback)")
    print("=" * 65)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    asl_classes: list[str] = ["A", "B", "C"] if args.test else ASL_CLASSES
    if args.test:
        print("\nQUICK TEST MODE — 3 ASL classes + all 5 emergency signs\n")

    all_classes = asl_classes + EMERGENCY_CLASSES

    # ── Locate ASL dataset ────────────────────────────────────────────────────
    asl_src: Path | None = None
    if args.data_dir:
        asl_src = _find_asl_source(Path(args.data_dir))
    elif args.skip_download:
        for candidate in [
            Path("data/raw_asl"),
            Path.home() / ".cache" / "kagglehub",
        ]:
            if candidate.is_dir():
                try:
                    asl_src = _find_asl_source(candidate)
                    print(f"Found existing dataset: {asl_src}")
                    break
                except FileNotFoundError:
                    pass
        if asl_src is None:
            print("ERROR: No existing dataset found.  Run without --skip-download.")
            sys.exit(1)
    else:
        raw = _download_asl()
        asl_src = _find_asl_source(raw)

    print(f"\nASL source: {asl_src}\n")

    # ── Extract real landmarks ─────────────────────────────────────────────
    print(f"Extracting landmarks for {len(asl_classes)} ASL classes …")
    X_asl, y_asl = extract_asl_landmarks(asl_src, asl_classes, args.max_per_class)
    if len(X_asl) == 0:
        print("ERROR: No samples collected.  Check your dataset path.")
        sys.exit(1)
    print(f"\nASL total: {len(X_asl)} samples\n")

    # ── Emergency signs: real dataset + synthetic fallback/supplement ───────
    n_asl = len(asl_classes)
    X_emg_rows: list[np.ndarray] = []
    y_emg_rows: list[int] = []
    emergency_src: Path | None = None

    if args.emergency_dir:
        src = Path(args.emergency_dir)
        X_emg, y_emg = extract_emergency_landmarks(src, GESTURE_TO_EMERGENCY_MAP, EMERGENCY_CLASSES, args.max_per_class)
        if len(X_emg) > 0:
            X_emg_rows.append(X_emg)
            y_emg_rows.extend((n_asl + y_emg).tolist())
            emergency_src = src
            print(f"Emergency source (--emergency-dir): {src}\n")

    if emergency_src is None and args.data_dir:
        src = Path(args.data_dir)
        X_emg, y_emg = extract_emergency_landmarks(src, GESTURE_TO_EMERGENCY_MAP, EMERGENCY_CLASSES, args.max_per_class)
        if len(X_emg) > 0:
            X_emg_rows.append(X_emg)
            y_emg_rows.extend((n_asl + y_emg).tolist())
            emergency_src = src
            print(f"Emergency source (from --data-dir): {src}\n")

    if emergency_src is None:
        raw_emg = _download_emergency_dataset()
        if raw_emg is not None:
            X_emg, y_emg = extract_emergency_landmarks(
                raw_emg, GESTURE_TO_EMERGENCY_MAP, EMERGENCY_CLASSES, args.max_per_class
            )
            if len(X_emg) > 0:
                X_emg_rows.append(X_emg)
                y_emg_rows.extend((n_asl + y_emg).tolist())
                emergency_src = raw_emg
                print(f"Emergency source (Kaggle download): {raw_emg}\n")

    print("Generating synthetic landmarks for emergency signs …")
    for i, sign in enumerate(EMERGENCY_CLASSES):
        class_idx = n_asl + i
        real_count = sum(1 for yv in y_emg_rows if yv == class_idx)
        if sign == "danger":
            synthetic_count = args.synthetic_per_sign
        elif real_count == 0:
            synthetic_count = args.synthetic_per_sign
        elif real_count < args.synthetic_per_sign // 2:
            synthetic_count = args.synthetic_per_sign - real_count
        else:
            synthetic_count = args.synthetic_per_sign // 3

        if synthetic_count > 0:
            samples = generate_synthetic_samples(_SYNTHETIC_SIGNS[sign], n_samples=synthetic_count)
            X_emg_rows.append(samples)
            y_emg_rows.extend([class_idx] * len(samples))
            reason = "supplement"
            if real_count == 0:
                reason = "no real images found"
            if sign == "danger":
                reason = "ilv gesture not in dataset"
            print(f"  [{sign}] {len(samples)} synthetic samples  ({reason})")

    X_emg = np.vstack(X_emg_rows)
    y_emg = np.array(y_emg_rows, dtype=np.int64)

    # ── Combine & split ────────────────────────────────────────────────────
    X = np.vstack([X_asl, X_emg])
    y = np.concatenate([y_asl, y_emg])
    print(f"\nFull dataset: {len(X)} samples | {X.shape[1]} features | {len(all_classes)} classes")
    print("  (63 coords + 19 geometric: tip distances, finger extensions, lateral spreads)\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # ── Train pipeline: Scaler → SVM ──────────────────────────────────────
    # PCA removed: was compressing 63D → 9D, discarding B/W discriminative
    # dimensions. 82 features is still small — RBF-SVM handles it directly.
    # C=50: tighter boundary than default C=10, better for similar signs.
    # class_weight='balanced': compensates if any class has fewer samples.
    print("Training pipeline: StandardScaler → SVM(RBF, C=50, balanced) …")
    print("(CPU only — ~10-15 min on Colab free tier with 1000 samples/class)\n")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=50,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=all_classes, zero_division=0))

    # ── Emergency sign accuracy breakdown ─────────────────────────────────
    print("Emergency sign breakdown:")
    for i, sign in enumerate(EMERGENCY_CLASSES):
        label_idx = n_asl + i
        mask = y_test == label_idx
        if mask.sum() == 0:
            continue
        sign_acc = accuracy_score(y_test[mask], y_pred[mask]) * 100
        print(f"  {sign:15s}: {sign_acc:5.1f}%")

    # ── Save ───────────────────────────────────────────────────────────────
    # The .joblib bundles Scaler + SVM weights together (no PCA).
    # Copy ONLY these 2 files to your target device — no dataset, no GPU.
    svm_path = MODEL_DIR / "sign_language_svm.joblib"
    cn_path = MODEL_DIR / "class_names_svm.json"
    joblib.dump(pipeline, svm_path)
    with open(cn_path, "w") as f:
        json.dump(all_classes, f, indent=2)

    print(f"\n✅ Saved model  → {svm_path}")
    print(f"✅ Saved labels → {cn_path}")
    print("\nThese 2 files are all you need on the target device.")
    print("Target device requirements: mediapipe, scikit-learn, opencv-python, joblib")
    print("  (NO dataset, NO GPU, NO internet after install)")


if __name__ == "__main__":
    main()
