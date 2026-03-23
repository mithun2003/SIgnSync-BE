# SignSync Model Training Report Snippets

All snippets below are taken from:
`src/app/core/ml/sign_model_pytorch/train_svm_standalone.py`

## 1) Core libraries and class configuration

```python
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
```

Explanation: This sets up the standalone SVM training stack (OpenCV + MediaPipe + scikit-learn), defines dataset constants, and fixes the final 33-class label space.

## 2) MediaPipe detector setup

```python
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
```

Explanation: The script auto-downloads `hand_landmarker.task` when missing and initializes a single-hand detector used throughout feature extraction.

## 3) Landmark extraction into the model feature vector

```python
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
```

Explanation: Each image is converted to an 82-D vector by wrist-centering, scale normalization, and appending geometry features so training and inference share the same representation.

## 4) Geometric feature computation

```python
_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
_MCPS = [2, 5, 9, 13, 17]  # thumb CMC + finger MCPs
_TIP_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]  # 10 pairs


def _compute_geometric_features(pts_norm: np.ndarray) -> np.ndarray:
    """Compute 19 geometric features from a (21, 3) normalized landmark array.

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
```

Explanation: This adds hand-shape structure (distances, extension, spread) that helps SVM separate look-alike classes better than raw coordinates alone.

## 5) Synthetic emergency-sign augmentation

```python
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
```

Explanation: For emergency classes with limited or missing real images, this function generates robust synthetic 82-D samples by perturbing canonical landmark poses.

## 6) ASL landmark extraction from dataset folders

```python
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
```

Explanation: This is the real-data ingestion stage that reads class folders, runs MediaPipe per image, and returns `(X, y)` arrays for model training.

## 7) Train/test split and SVM pipeline

```python
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
```

Explanation: The final feature matrix is stratified into train/test, then trained with a `StandardScaler + RBF SVC` pipeline using balanced class weights.

## 8) Evaluation and model artifact saving

```python
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
```

Explanation: The script reports overall and per-emergency-class metrics, then saves the complete inference artifacts (`.joblib` model + class-name JSON).
