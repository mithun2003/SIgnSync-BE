#!/usr/bin/env python3
"""
Sign Language SVM Trainer  (Primary Model)
==========================================

Trains an SVM on MediaPipe hand landmarks.  More accurate than CNN for this
task because it works on clean, normalised 3-D joint coordinates instead of
raw pixels — no background clutter, no lighting variation.

Classes
-------
  • 26 ASL alphabets (A-Z)          — real landmarks from Kaggle dataset
  • space + del                      — real landmarks from Kaggle dataset
  • help / danger / emergency        — SYNTHETIC landmarks (no webcam needed)

Total: 31 classes

Usage
-----
  python train_svm.py               # full training (downloads ASL dataset)
  python train_svm.py --test        # quick smoke-test (3 ASL classes)
  python train_svm.py --skip-download   # dataset already present
  python train_svm.py --data-dir /path/to/asl_alphabet_train

Output
------
  trained_model/
    sign_language_svm.joblib    – trained pipeline (Scaler → PCA → SVM)
    class_names_svm.json        – ordered class labels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from config import (
    ASL_CLASSES,
    DATA_DIR,
    EMERGENCY_CLASSES,
    KAGGLE_DATASET,
    MODEL_DIR,
    RANDOM_SEED,
    SKELETON_DIR,
)
from utils import extract_landmarks_array

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic emergency sign landmark definitions
# ─────────────────────────────────────────────────────────────────────────────
#
# Coordinates are in the SAME normalised space as extract_landmarks_array:
#   • Wrist (landmark 0) is subtracted  →  wrist always at (0, 0, 0)
#   • Divided by wrist-to-middle-MCP distance
#   • y is NEGATIVE upward (image origin = top-left, y grows downward)
#
# MediaPipe landmark order (21 × 3):
#   0:  WRIST
#   1-4:  THUMB  (CMC, MCP, IP, TIP)
#   5-8:  INDEX  (MCP, PIP, DIP, TIP)
#   9-12: MIDDLE (MCP, PIP, DIP, TIP)  ← middle-MCP defines the scale (≈ -1.0 y)
#   13-16:RING   (MCP, PIP, DIP, TIP)
#   17-20:PINKY  (MCP, PIP, DIP, TIP)
#
# Each emergency sign is designed to be VISUALLY MEMORABLE and DISTINCT
# from all 26 ASL letter handshapes.

_SYNTHETIC_SIGNS: dict[str, np.ndarray] = {
    # ── HELP: Open flat palm, all 5 fingers fully spread ─────────────────────
    # Universal "stop / help" gesture.  Maximum finger extension — highly distinct.
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
    # "I Love You" handshape repurposed as a distress signal.
    # NOT present in any of the 26 ASL letter shapes — unique combination.
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
            [0.02, -0.79, 0.140],  # middle PIP  (curled INTO palm — positive z)
            [-0.02, -0.57, 0.190],  # middle DIP
            [-0.04, -0.42, 0.210],  # middle TIP  (touching palm)
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
    # Strong, globally understood emergency signal.
    # Key difference from ASL 'A': thumb points straight UP, not sideways.
    "emergency": np.array(
        [
            [0.00, 0.00, 0.000],  # wrist
            [0.20, -0.20, 0.000],  # thumb CMC  (starts going up)
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
}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_synthetic_samples(
    base_pose: np.ndarray,
    n_samples: int = 600,
    noise_std: float = 0.055,
) -> np.ndarray:
    """Augment a canonical hand pose into *n_samples* realistic vectors.

    Augmentations:
      1. Gaussian noise       — simulates MediaPipe jitter
      2. Random XY rotation ±20°  — tilted hand
      3. Random scale ±15%    — varying camera distance
      4. X-axis mirror (every other sample)  — left-hand users

    Returns ``(n_samples, 63)`` float32 array.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    samples: list[np.ndarray] = []

    for i in range(n_samples):
        noisy = base_pose + rng.normal(0, noise_std, base_pose.shape).astype(np.float32)

        angle = rng.uniform(-np.pi / 9, np.pi / 9)
        ca, sa = float(np.cos(angle)), float(np.sin(angle))
        rotated = noisy.copy()
        rotated[:, 0] = noisy[:, 0] * ca - noisy[:, 1] * sa
        rotated[:, 1] = noisy[:, 0] * sa + noisy[:, 1] * ca

        rotated *= float(rng.uniform(0.85, 1.15))

        if i % 2 == 1:
            rotated[:, 0] = -rotated[:, 0]

        samples.append(rotated.flatten())

    return np.array(samples, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ASL dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _find_asl_source(root: Path) -> Path:
    for c in [
        root / "asl_alphabet_train" / "asl_alphabet_train",
        root / "asl_alphabet_train",
        root,
    ]:
        if c.is_dir() and ((c / "A").exists() or (c / "a").exists()):
            return c
    for d in root.rglob("A"):
        if d.is_dir():
            return d.parent
    raise FileNotFoundError(f"Cannot find ASL class folders in {root}")


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
        print("Tip: add ~/.kaggle/kaggle.json  OR  pass --data-dir")
        sys.exit(1)


def extract_asl_landmarks(
    src_dir: Path,
    classes: list[str],
    max_per_class: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract MediaPipe landmark vectors from raw ASL image folder."""
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    for cls_idx, cls in enumerate(classes):
        src_cls = None
        for v in [cls, cls.upper(), cls.lower()]:
            d = src_dir / v
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

        print(f"  [{cls}] {ok} landmarks extracted  ({fail} no-hand)")

    if not X_rows:
        return np.empty((0, 63), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sign language SVM (primary model)")
    parser.add_argument("--test", action="store_true", help="Quick test: only first 3 ASL classes")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip Kaggle download; search for existing dataset"
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to raw ASL dataset folder (skips download)")
    parser.add_argument("--max-per-class", type=int, default=500, help="Max real images per ASL class (default: 500)")
    parser.add_argument(
        "--synthetic-per-sign", type=int, default=600, help="Synthetic samples per emergency sign (default: 600)"
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    asl_classes: list[str] = ["A", "B", "C"] if args.test else ASL_CLASSES
    if args.test:
        print("QUICK TEST MODE — 3 ASL classes + all 3 emergency signs")

    all_classes = asl_classes + EMERGENCY_CLASSES

    # ── Locate ASL raw images ─────────────────────────────────────────────────
    asl_src: Path | None = None
    if args.data_dir:
        asl_src = _find_asl_source(Path(args.data_dir))
    elif args.skip_download:
        # Try known cache locations
        candidates = [
            DATA_DIR / "raw_asl",
            SKELETON_DIR,  # skeleton images also work for landmark extraction
            Path.home() / ".cache" / "kagglehub",
        ]
        for c in candidates:
            if c.is_dir():
                try:
                    asl_src = _find_asl_source(c)
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

    # ── Extract real landmarks (ASL) ──────────────────────────────────────────
    print(f"Extracting landmarks for {len(asl_classes)} ASL classes …")
    X_asl, y_asl = extract_asl_landmarks(asl_src, asl_classes, args.max_per_class)
    if len(X_asl) == 0:
        print("ERROR: No samples collected.  Check your dataset path.")
        sys.exit(1)
    print(f"ASL total: {len(X_asl)} samples\n")

    # ── Generate synthetic landmarks (emergency signs) ────────────────────────
    print("Generating synthetic landmarks for emergency signs …")
    X_emg_rows: list[np.ndarray] = []
    y_emg_rows: list[int] = []
    n_asl = len(asl_classes)

    for i, sign in enumerate(EMERGENCY_CLASSES):
        samples = generate_synthetic_samples(_SYNTHETIC_SIGNS[sign], n_samples=args.synthetic_per_sign)
        X_emg_rows.append(samples)
        y_emg_rows.extend([n_asl + i] * len(samples))
        print(f"  [{sign}] {len(samples)} synthetic samples")

    X_emg = np.vstack(X_emg_rows)
    y_emg = np.array(y_emg_rows, dtype=np.int64)

    # ── Combine ───────────────────────────────────────────────────────────────
    X = np.vstack([X_asl, X_emg])
    y = np.concatenate([y_asl, y_emg])
    print(f"\nFull dataset: {len(X)} samples, {X.shape[1]} features, {len(all_classes)} classes")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Fitting: StandardScaler → PCA(95 %) → SVM(RBF, C=10) …")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=RANDOM_SEED)),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_SEED)),
        ]
    )
    pipeline.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=all_classes, zero_division=0))

    # ── Save ──────────────────────────────────────────────────────────────────
    svm_path = MODEL_DIR / "sign_language_svm.joblib"
    joblib.dump(pipeline, svm_path)
    cn_path = MODEL_DIR / "class_names_svm.json"
    with open(cn_path, "w") as f:
        json.dump(all_classes, f, indent=2)

    print(f"\nSaved model  → {svm_path}")
    print(f"Saved labels → {cn_path}")
    print("\nTest it: python predict.py --webcam")


if __name__ == "__main__":
    main()
