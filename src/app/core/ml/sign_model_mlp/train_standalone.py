#!/usr/bin/env python3
"""
MLP Sign Language Model — Standalone Colab Training Script
===========================================================
Single file: no config imports, no external dependencies beyond what's pip-installed.

COLAB SETUP
-----------
1. Open Google Colab (free CPU runtime is enough — ~10 min training)
2. Mount Drive or just let it run — model files are saved locally then download.
3. Install deps:

    !pip install torch torchvision mediapipe scikit-learn tqdm onnxruntime kaggle

4. Upload kaggle.json to Colab:
    from google.colab import files
    files.upload()    # select your kaggle.json
    !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

5. Run this script:
    !python train_mlp_standalone.py

6. Download the trained files from  trained_model/  folder.
   Upload to backend:  src/app/core/ml/sign_model_mlp/trained_model/

WHAT THIS TRAINS
----------------
MediaPipe 63-D hand landmarks  →  MLP  →  33 ASL classes
  • A-Z (26)  loaded from Kaggle asl-alphabet dataset
  • space, del loaded from Kaggle dataset
  • help, danger, emergency, thumbs_down, ok_sign — synthetic landmarks (no collection!)

Saves:
  trained_model/sign_language_mlp.pt       — PyTorch weights (for fine-tuning)
  trained_model/sign_language_mlp.onnx     — ONNX model (backend runtime)
  trained_model/class_names_mlp.json       — label list

TO SWITCH BACKEND FROM SVM → MLP
---------------------------------
In  src/app/core/ml/predict.py  change the two import lines at the top:

    # from .sign_model_pytorch.predict_svm import (...)   # SVM
    from .sign_model_mlp.predict import (                 # MLP
        load_ml_model, predict_sign, predict_sign_from_landmarks,
        get_public_info, get_health_status, reload_model,
    )
"""

# ── stdlib / third-party ──────────────────────────────────────────────────────
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Classes — must stay in this order (indices baked into model + JSON)
ASL_CLASSES: list[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del"]
EMERGENCY_CLASSES: list[str] = ["help", "danger", "emergency", "thumbs_down", "ok_sign"]
ALL_CLASSES: list[str] = ASL_CLASSES + EMERGENCY_CLASSES  # 33 total

INPUT_DIM = 63  # 21 landmarks × 3 (x, y, z)
HIDDEN1 = 256
HIDDEN2 = 128
NUM_CLASSES = len(ALL_CLASSES)

BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.20
RANDOM_SEED = 42
SYNTHETIC_VARIATIONS = 500  # variations per emergency sign
IMAGES_PER_LETTER = 500  # 500 is enough for SVM on landmarks (~95-96% accuracy)

KAGGLE_DATASET = "grassknoted/asl-alphabet"
DATA_DIR = Path("data/asl_alphabet_train")
OUT_DIR = Path("trained_model")
OUT_DIR.mkdir(exist_ok=True)

# MediaPipe Tasks API (replaces deprecated mp.solutions.hands)
from mediapipe.tasks import python as _mp_tasks  # noqa: E402
from mediapipe.tasks.python import vision as _mp_vision  # noqa: E402

_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
try:
    _TASK_MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
except NameError:
    # Running inside Colab / Jupyter where __file__ is not defined
    _TASK_MODEL_PATH = Path.cwd() / "hand_landmarker.task"
_detector = None


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
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        _detector = _mp_vision.HandLandmarker.create_from_options(opts)
    return _detector


# ═══════════════════════════════════════════════════════════════════════════════
# ── MODEL DEFINITION ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


class SignLanguageMLP(nn.Module):
    """63→256→128→33  (~85 K parameters)"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN1),
            nn.BatchNorm1d(HIDDEN1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.BatchNorm1d(HIDDEN2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN2, NUM_CLASSES),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# ── LANDMARK HELPERS ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def extract_landmarks(image_path: Path) -> np.ndarray | None:
    """Extract + normalize 63-D landmark vector from an image file."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = _get_detector().detect(mp_img)
    if not result.hand_landmarks:
        return None
    return _normalize_mp_lms(result.hand_landmarks[0])


def _normalize_mp_lms(lms) -> np.ndarray:
    """Subtract wrist and scale by wrist→middle-MCP distance."""
    wrist = lms[0]
    coords: list[float] = []
    for lm in lms:
        coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    arr = np.array(coords, dtype=np.float32)
    mid = lms[9]
    scale = float(np.sqrt((mid.x - wrist.x) ** 2 + (mid.y - wrist.y) ** 2))
    if scale > 1e-6:
        arr /= scale
    return arr


def _normalize_raw(arr: np.ndarray) -> np.ndarray:
    """Same normalization applied to a flat 63-element array."""
    arr = arr.copy().astype(np.float32)
    wrist = arr[:3].copy()
    for i in range(21):
        arr[i * 3 : i * 3 + 3] -= wrist
    scale = float(np.sqrt(arr[27] ** 2 + arr[28] ** 2))  # landmark 9 = index 27-29
    if scale > 1e-6:
        arr /= scale
    return arr


def generate_variations(base: np.ndarray, n: int = SYNTHETIC_VARIATIONS) -> np.ndarray:
    """Augment a base landmark array with noise + scale perturbation."""
    out = []
    for _ in range(n):
        v = (base + np.random.normal(0, 0.02, base.shape)) * np.random.uniform(0.92, 1.08)
        out.append(_normalize_raw(v))
    return np.array(out, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# ── EMERGENCY SIGN DEFINITIONS ────────────────────────────────────────────────
# (same canonical landmarks used in SVM training — must stay consistent)
# ═══════════════════════════════════════════════════════════════════════════════


def _base_landmarks() -> dict[str, np.ndarray]:
    return {
        "help": np.array(
            [
                0.5,
                0.9,
                0.0,
                0.4,
                0.8,
                -0.02,
                0.35,
                0.7,
                -0.03,
                0.32,
                0.6,
                -0.04,
                0.3,
                0.5,
                -0.05,
                0.45,
                0.75,
                0.0,
                0.45,
                0.55,
                0.01,
                0.45,
                0.35,
                0.02,
                0.45,
                0.15,
                0.03,
                0.5,
                0.75,
                0.0,
                0.5,
                0.5,
                0.01,
                0.5,
                0.3,
                0.02,
                0.5,
                0.1,
                0.03,
                0.55,
                0.75,
                0.0,
                0.55,
                0.55,
                0.01,
                0.55,
                0.35,
                0.02,
                0.55,
                0.15,
                0.03,
                0.6,
                0.8,
                0.0,
                0.6,
                0.6,
                0.01,
                0.6,
                0.4,
                0.02,
                0.6,
                0.2,
                0.03,
            ],
            dtype=np.float32,
        ),
        "danger": np.array(
            [
                0.5,
                0.9,
                0.0,
                0.4,
                0.7,
                -0.02,
                0.35,
                0.65,
                -0.03,
                0.3,
                0.6,
                -0.04,
                0.28,
                0.55,
                -0.05,
                0.45,
                0.75,
                0.0,
                0.45,
                0.55,
                0.01,
                0.45,
                0.35,
                0.02,
                0.45,
                0.15,
                0.03,
                0.5,
                0.75,
                0.0,
                0.52,
                0.68,
                0.02,
                0.54,
                0.63,
                0.04,
                0.56,
                0.6,
                0.06,
                0.55,
                0.75,
                0.0,
                0.56,
                0.68,
                0.02,
                0.57,
                0.63,
                0.04,
                0.58,
                0.6,
                0.06,
                0.58,
                0.72,
                0.0,
                0.58,
                0.55,
                0.01,
                0.58,
                0.38,
                0.02,
                0.58,
                0.2,
                0.03,
            ],
            dtype=np.float32,
        ),
        "emergency": np.array(
            [
                0.5,
                0.8,
                0.0,
                0.45,
                0.65,
                -0.02,
                0.42,
                0.55,
                -0.03,
                0.4,
                0.48,
                -0.04,
                0.38,
                0.42,
                0.0,
                0.48,
                0.7,
                0.0,
                0.5,
                0.65,
                0.02,
                0.52,
                0.62,
                0.04,
                0.54,
                0.6,
                0.06,
                0.52,
                0.7,
                0.0,
                0.54,
                0.65,
                0.02,
                0.56,
                0.62,
                0.04,
                0.58,
                0.6,
                0.06,
                0.56,
                0.7,
                0.0,
                0.57,
                0.65,
                0.02,
                0.58,
                0.62,
                0.04,
                0.59,
                0.6,
                0.06,
                0.59,
                0.72,
                0.0,
                0.6,
                0.67,
                0.02,
                0.61,
                0.64,
                0.04,
                0.62,
                0.62,
                0.06,
            ],
            dtype=np.float32,
        ),
        "thumbs_down": np.array(
            [
                0.5,
                0.5,
                0.0,
                0.4,
                0.55,
                -0.02,
                0.35,
                0.62,
                -0.03,
                0.3,
                0.7,
                -0.04,
                0.27,
                0.78,
                -0.05,
                0.48,
                0.45,
                0.0,
                0.5,
                0.4,
                0.02,
                0.52,
                0.38,
                0.04,
                0.54,
                0.37,
                0.06,
                0.52,
                0.45,
                0.0,
                0.54,
                0.4,
                0.02,
                0.56,
                0.38,
                0.04,
                0.58,
                0.37,
                0.06,
                0.56,
                0.45,
                0.0,
                0.57,
                0.4,
                0.02,
                0.58,
                0.38,
                0.04,
                0.59,
                0.37,
                0.06,
                0.59,
                0.47,
                0.0,
                0.6,
                0.42,
                0.02,
                0.61,
                0.39,
                0.04,
                0.62,
                0.37,
                0.06,
            ],
            dtype=np.float32,
        ),
        "ok_sign": np.array(
            [
                0.5,
                0.8,
                0.0,
                0.42,
                0.72,
                -0.02,
                0.38,
                0.65,
                -0.03,
                0.42,
                0.6,
                -0.04,
                0.46,
                0.65,
                -0.03,
                0.45,
                0.65,
                0.0,
                0.44,
                0.52,
                0.01,
                0.44,
                0.4,
                0.02,
                0.44,
                0.28,
                0.03,
                0.5,
                0.62,
                0.0,
                0.52,
                0.55,
                0.02,
                0.54,
                0.5,
                0.04,
                0.56,
                0.47,
                0.06,
                0.55,
                0.65,
                0.0,
                0.56,
                0.58,
                0.02,
                0.57,
                0.52,
                0.04,
                0.58,
                0.48,
                0.06,
                0.58,
                0.68,
                0.0,
                0.6,
                0.61,
                0.02,
                0.61,
                0.55,
                0.04,
                0.62,
                0.51,
                0.06,
            ],
            dtype=np.float32,
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ── DATA LOADING ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def load_alphabet_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load A-Z + space + del from Kaggle dataset folders."""
    X, y = [], []
    folder_map = {
        **{letter: i for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")},
        "space": 26,
        "del": 27,
    }
    for folder, class_idx in folder_map.items():
        folder_dir = data_dir / folder
        if not folder_dir.exists():
            print(f"⚠️  Folder not found: {folder_dir}")
            continue
        images = list(folder_dir.glob("*.jpg"))[:IMAGES_PER_LETTER]
        for img_path in tqdm(images, desc=f"  {folder}", leave=False):
            feat = extract_landmarks(img_path)
            if feat is not None:
                X.append(feat)
                y.append(class_idx)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def build_full_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Combine alphabet data + synthetic emergency signs."""
    print("\n📂 Loading ASL alphabet (A-Z + space + del)…")
    X_alph, y_alph = load_alphabet_data(data_dir)
    print(f"   ✅ Alphabet: {len(X_alph)} samples")

    bases = _base_landmarks()
    X_parts = [X_alph]
    y_parts = [y_alph]

    print("\n🔧 Generating synthetic emergency signs…")
    for i, name in enumerate(EMERGENCY_CLASSES):
        base = _normalize_raw(bases[name])
        vars_ = generate_variations(base, SYNTHETIC_VARIATIONS)
        label = 28 + i  # indices 28-32
        X_parts.append(vars_)
        y_parts.append(np.full(SYNTHETIC_VARIATIONS, label, dtype=np.int64))
        print(f"   ✅ {name}: {SYNTHETIC_VARIATIONS} synthetic samples")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    print(f"\n📊 Total: {len(X)} samples | {NUM_CLASSES} classes")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# ── TRAINING ──────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def train_model(X: np.ndarray, y: np.ndarray) -> SignLanguageMLP:
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    dataset = TensorDataset(X_t, y_t)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SignLanguageMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_state = None

    print(f"\n🚀 Training MLP for {EPOCHS} epochs…")
    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        train_correct = train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            preds = model(xb).argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)
        scheduler.step()

        # ── validate ──
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        t_acc = train_correct / train_total * 100
        v_acc = val_correct / val_total * 100

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}  train={t_acc:.1f}%  val={v_acc:.1f}%"
                + ("  ★ best" if v_acc == best_val_acc else "")
            )

    print(f"\n✅ Best validation accuracy: {best_val_acc:.2f}%")
    model.load_state_dict(best_state)
    return model.cpu()


# ═══════════════════════════════════════════════════════════════════════════════
# ── EVALUATION ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_model(model: SignLanguageMLP, X: np.ndarray, y: np.ndarray) -> None:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X))
    preds = logits.argmax(dim=1).numpy()

    # Emergency sign accuracy
    print("\n🚨 Emergency Sign Accuracy:")
    for i, name in enumerate(EMERGENCY_CLASSES):
        idx = 28 + i
        mask = y == idx
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == y[mask]).mean() * 100
        print(f"   {name:15s}: {acc:.1f}%")

    print("\n📋 Full Classification Report:")
    print(classification_report(y, preds, target_names=ALL_CLASSES, zero_division=0))


# ═══════════════════════════════════════════════════════════════════════════════
# ── EXPORT ────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def export_model(model: SignLanguageMLP) -> None:
    # Save PyTorch checkpoint
    pt_path = OUT_DIR / "sign_language_mlp.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"\n💾 PyTorch weights → {pt_path}")

    # Export to ONNX using classic trace-based exporter (no onnxscript needed)
    onnx_path = OUT_DIR / "sign_language_mlp.onnx"
    dummy = torch.zeros(1, INPUT_DIM)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,  # force classic trace-based export — no onnxscript needed
        )
    print(f"📦 ONNX model       → {onnx_path}")

    # Save labels
    labels_path = OUT_DIR / "class_names_mlp.json"
    with open(labels_path, "w") as f:
        json.dump(ALL_CLASSES, f, indent=2)
    print(f"🏷️  Class names      → {labels_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ── KAGGLE DOWNLOAD ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def maybe_download_dataset() -> None:
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"✅ Dataset found at {DATA_DIR}")
        return

    print("📥 Downloading Kaggle dataset via kagglehub…")
    try:
        import shutil

        import kagglehub

        # kagglehub downloads to a cache dir; we copy to our DATA_DIR
        cache_path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"   Cache path: {cache_path}")

        # Find the asl_alphabet_train subfolder
        src = Path(cache_path)
        train_dir = next(
            (p for p in src.rglob("asl_alphabet_train") if p.is_dir()),
            None,
        )
        if train_dir is None:
            # Fallback: use the downloaded root directly
            train_dir = src

        DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        if not DATA_DIR.exists():
            shutil.copytree(train_dir, DATA_DIR)
        print(f"✅ Dataset ready at {DATA_DIR}")

    except Exception as e:
        raise RuntimeError(
            f"Kaggle download failed: {e}\n"
            "Make sure you ran Step 2 (kaggle.json upload) before this cell.\n"
            "Check: !ls ~/.kaggle/kaggle.json"
        ) from e


# ═══════════════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point — reads module-level config constants.

    To override from a Colab notebook, set attributes on the module before calling main():     module.IMAGES_PER_LETTER
    = 500     module.EPOCHS = 60     module.main()
    """
    print("=" * 70)
    print("SignSync — MLP Training (MediaPipe Landmarks)")
    print("=" * 70)
    print(f"  Classes: {NUM_CLASSES}  |  Input: {INPUT_DIM}-D  |  Arch: {INPUT_DIM}→{HIDDEN1}→{HIDDEN2}→{NUM_CLASSES}")

    maybe_download_dataset()

    X, y = build_full_dataset(DATA_DIR)
    model = train_model(X, y)
    evaluate_model(model, X, y)
    export_model(model)

    print("\n" + "=" * 70)
    print("DONE — copy trained_model/ contents to:")
    print("  backend/src/app/core/ml/sign_model_mlp/trained_model/")
    print("=" * 70)


if __name__ == "__main__":
    main()
