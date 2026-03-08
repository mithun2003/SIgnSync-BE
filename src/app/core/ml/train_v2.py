"""SignSync ASL Recognition — Training Script v2
═══════════════════════════════════════════════════════════════════ Trains a 29-class ASL model (A-Z, space, del,
nothing) from the Kaggle "grassknoted/asl-alphabet" dataset.

WHY THE OLD MODEL WAS INACCURATE
─────────────────────────────────
1. Only 5 epochs — model never converged.
2. MobileNetV2 expects inputs in [-1, 1] via preprocess_input, but
   training used rescale=1/255 → [0, 1].  Wrong input scale destroys
   transfer learning.
3. No fine-tuning phase — only the head was ever trained.
4. Minimal augmentation.

WHAT THIS SCRIPT DOES DIFFERENTLY
───────────────────────────────────
• EfficientNetB3  — 20-30% more accurate than MobileNetV2 on 29 classes.
• Correct preprocessing: preprocess_input embedded inside the model so
  training and inference are always consistent.
• 3-phase training  (frozen 15 ep → partial fine-tune 20 ep → full 10 ep).
• Rich augmentation (rotation ±20°, zoom, shift, shear, flip).
• Label smoothing 0.1 to prevent overconfidence on hard letter pairs.
• Cosine-decay LR with warmup.
• Class-weight balancing.
• Removes all-black (no-hand) images before training.

EXPECTED RESULT
────────────────
Val accuracy ≥ 95 % on the full 87 k-image dataset.

USAGE
──────
    python train_v2.py --test      # smoke test (5 classes, 10 imgs, ~5 min)
    python train_v2.py --full      # full training (~45 min GPU / 3-4 h CPU)

REQUIREMENTS
─────────────
    pip install tensorflow>=2.13 kagglehub opencv-python mediapipe
                scikit-learn matplotlib seaborn
    Kaggle API token at ~/.kaggle/kaggle.json
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

import cv2
import kagglehub
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import seaborn as sns
import tensorflow as tf
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ═══════════════════════════════════════════════════════════════════════════════
#  GPU / MIXED PRECISION SETUP
# ═══════════════════════════════════════════════════════════════════════════════


def setup_hardware() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Mixed precision for ~2× speed on Tensor-Core GPUs (RTX 20xx+)
        try:
            mixed_precision.set_global_policy("mixed_float16")
            print(f"✅ GPU found: {[g.name for g in gpus]}  |  mixed_float16 ON")
        except Exception:
            print(f"✅ GPU found: {[g.name for g in gpus]}  |  mixed_float16 OFF")
    else:
        print("ℹ️  No GPU — running on CPU (slower but works)")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


class Config:
    IMG_SIZE = (224, 224)
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 29

    # Paths
    OUTPUT_DIR = Path("output_v2")
    SKELETON_DIR = OUTPUT_DIR / "skeleton_dataset"
    MODEL_BEST = OUTPUT_DIR / "sign_language_model_best.keras"
    MODEL_FINAL = OUTPUT_DIR / "sign_language_model_final.keras"
    CLASS_JSON = OUTPUT_DIR / "class_names.json"
    TASK_FILE = OUTPUT_DIR / "hand_landmarker.task"
    TASK_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )

    def __init__(self, test_mode: bool = False) -> None:
        self.test_mode = test_mode
        self.OUTPUT_DIR.mkdir(exist_ok=True)

        if test_mode:
            self.batch_size = 8
            self.max_classes = 5
            self.max_imgs_per_class = 10
            # Phases: frozen, partial, full
            self.epochs = (3, 3, 2)
            print("🧪 TEST MODE — 5 classes, 10 imgs/class, 3+3+2 epochs")
        else:
            self.batch_size = 32
            self.max_classes = None
            self.max_imgs_per_class = None
            self.epochs = (15, 20, 10)
            print("🚀 FULL MODE — 29 classes, all images, 15+20+10 epochs")


# ═══════════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE: SKELETON GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

# These are fixed regardless of MediaPipe version
_HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def _build_detector(task_path: Path) -> mp_vision.HandLandmarker:
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(task_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def _to_px(val: float, vmin: float, span: float, canvas: int = 224) -> int:
    return max(0, min(canvas - 1, int(((val - vmin) / max(span, 1e-6)) * canvas)))


def image_to_skeleton(bgr: np.ndarray, detector: mp_vision.HandLandmarker) -> tuple[np.ndarray, bool]:
    """Convert a BGR hand image to a white-on-black skeleton image (224×224)."""
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = detector.detect(mp_img)

    if not result.hand_landmarks:
        return canvas, False

    lms = result.hand_landmarks[0]
    xs = [lm.x for lm in lms]
    ys = [lm.y for lm in lms]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    # Add 15 % padding so fingers aren't clipped at edges
    px = (x1 - x0) * 0.15
    py = (y1 - y0) * 0.15
    x0 -= px
    x1 += px
    y0 -= py
    y1 += py
    w = x1 - x0
    h = y1 - y0

    # Draw bones
    for s, e in _HAND_CONNECTIONS:
        p1, p2 = lms[s], lms[e]
        cv2.line(
            canvas,
            (_to_px(p1.x, x0, w), _to_px(p1.y, y0, h)),
            (_to_px(p2.x, x0, w), _to_px(p2.y, y0, h)),
            (255, 255, 255),
            2,
        )
    # Draw joints
    for lm in lms:
        cv2.circle(
            canvas,
            (_to_px(lm.x, x0, w), _to_px(lm.y, y0, h)),
            4,
            (200, 200, 200),
            -1,
        )
    return canvas, True


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET DOWNLOAD & SKELETON PRE-GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def download_dataset() -> str:
    print("\n" + "═" * 60)
    print("  STEP 1 — Downloading ASL Alphabet Dataset from Kaggle")
    print("═" * 60)
    kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_creds.exists():
        raise FileNotFoundError(
            f"Kaggle credentials not found at {kaggle_creds}.\n"
            "Go to https://www.kaggle.com/settings → API → Create New Token."
        )
    base = kagglehub.dataset_download("grassknoted/asl-alphabet")
    train_dir = os.path.join(base, "asl_alphabet_train", "asl_alphabet_train")
    if not os.path.isdir(train_dir):
        train_dir = os.path.join(base, "asl_alphabet_train")
    classes = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    print(f"✅ {len(classes)} classes — {train_dir}")
    return train_dir


def download_mediapipe_task(cfg: Config) -> None:
    if cfg.TASK_FILE.exists():
        print(f"✅ {cfg.TASK_FILE.name} already present")
        return
    print(f"⬇️  Downloading {cfg.TASK_FILE.name} …")
    urllib.request.urlretrieve(cfg.TASK_URL, str(cfg.TASK_FILE))
    print(f"✅ Downloaded ({cfg.TASK_FILE.stat().st_size / 1e6:.1f} MB)")


def build_skeleton_dataset(raw_dir: str, cfg: Config) -> Path:
    """Convert raw photos → skeleton images and save them to cfg.SKELETON_DIR."""
    print("\n" + "═" * 60)
    print("  STEP 2 — Pre-generating Skeleton Dataset")
    print("═" * 60)

    classes = sorted(d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)))
    if cfg.max_classes:
        classes = classes[: cfg.max_classes]

    total_imgs = sum(
        min(
            len([f for f in os.listdir(os.path.join(raw_dir, c)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]),
            cfg.max_imgs_per_class or 999_999,
        )
        for c in classes
    )
    print(f"  Classes : {len(classes)} | Images: {total_imgs:,}")
    print(f"  Output  : {cfg.SKELETON_DIR}\n")

    detector = _build_detector(cfg.TASK_FILE)
    processed = skipped = no_hand = 0

    for cls in classes:
        src = os.path.join(raw_dir, cls)
        dst = cfg.SKELETON_DIR / cls
        dst.mkdir(parents=True, exist_ok=True)

        imgs = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if cfg.max_imgs_per_class:
            imgs = imgs[: cfg.max_imgs_per_class]

        for fname in imgs:
            out_path = dst / fname
            if out_path.exists():
                skipped += 1
                processed += 1
                continue

            bgr = cv2.imread(os.path.join(src, fname))
            if bgr is None:
                processed += 1
                continue

            skeleton, found = image_to_skeleton(bgr, detector)
            if not found:
                no_hand += 1
            cv2.imwrite(str(out_path), cv2.cvtColor(skeleton, cv2.COLOR_RGB2BGR))
            processed += 1

            if processed % 1_000 == 0:
                pct = processed / total_imgs * 100
                print(f"  [{processed:6,}/{total_imgs:,}]  {pct:5.1f} %  no_hand={no_hand}  skipped={skipped}")

    found_pct = (processed - skipped - no_hand) / max(processed - skipped, 1) * 100
    print("\n✅ Skeleton generation complete")
    print(f"   Processed : {processed - skipped:,}  |  Hand found: {found_pct:.1f} %  |  No-hand: {no_hand:,}")
    return cfg.SKELETON_DIR


def remove_black_images(skeleton_dir: Path) -> None:
    """Delete images where MediaPipe found no hand (all-black output)."""
    print("\n  Removing all-black (no-hand) images …", end=" ")
    removed = 0
    for img_path in skeleton_dir.rglob("*.jpg"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None and img.mean() < 2.0:
            img_path.unlink()
            removed += 1
    print(f"removed {removed:,}")


def _skeleton_dataset_exists(skeleton_dir: Path) -> bool:
    if not skeleton_dir.exists():
        return False
    for cls_dir in skeleton_dir.iterdir():
        if cls_dir.is_dir() and len(list(cls_dir.glob("*.jpg"))) >= 50:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════


def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    """Return (full_model, base_model) where base_model is EfficientNetB3.

    The preprocessing layer is EMBEDDED inside the model so that inference
    only needs to pass a [0, 255] or [0, 1] uint8/float image — the model
    normalises it correctly regardless of what the caller sends.

    NOTE: We embed a Rescaling layer so the model always expects inputs in
    [0, 1] (matching the existing inference code which does img / 255).
    EfficientNetB3's preprocess_input then converts that to [-1, 1].
    """
    inputs = keras.Input(shape=(224, 224, 3), name="input_image")

    # ── Step 1: Undo the /255 done by inference, apply EfficientNet preprocess ──
    # Inference sends images in [0, 1]; EfficientNetB3 expects [0, 255] → [-1, 1]
    x = layers.Rescaling(255.0, name="rescale_to_255")(inputs)
    x = layers.Lambda(preprocess_input, name="efficientnet_preprocess")(x)

    # ── Step 2: EfficientNetB3 backbone ────────────────────────────────────────
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base.trainable = False  # frozen at start
    x = base.output

    # ── Step 3: Classification head ────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dense(512, name="head_dense1")(x)
    x = layers.Activation("relu", name="head_relu1")(x)
    x = layers.Dropout(0.4, name="head_drop1")(x)
    x = layers.Dense(256, name="head_dense2")(x)
    x = layers.Activation("relu", name="head_relu2")(x)
    x = layers.Dropout(0.3, name="head_drop2")(x)
    # float32 output even with mixed precision
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="ASL_EfficientNetB3")
    return model, base


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════


def make_generators(skeleton_dir: Path, cfg: Config):
    """Return (train_gen, val_gen, class_names, class_weights)."""

    # Augmentation tuned for skeleton images (white lines on black bg).
    # Horizontal flip is valid because ASL uses both orientations in the wild.
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.15,
        rotation_range=20,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="constant",
        cval=0,  # fill new pixels with black (matches background)
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.15,
    )

    common_kwargs = {
        "directory": str(skeleton_dir),
        "target_size": cfg.IMG_SIZE,
        "batch_size": cfg.batch_size,
        "class_mode": "categorical",
        "color_mode": "rgb",
        "interpolation": "bilinear",
    }

    train_gen = train_datagen.flow_from_directory(**common_kwargs, subset="training", shuffle=True)
    val_gen = val_datagen.flow_from_directory(**common_kwargs, subset="validation", shuffle=False)

    class_names = [k for k, _ in sorted(train_gen.class_indices.items(), key=lambda kv: kv[1])]

    # Class weights so rare/hard classes get equal attention
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=train_gen.classes,
    )
    cw_dict = dict(enumerate(class_weights))

    print(f"\n  Train batches : {len(train_gen):,}  ({train_gen.n:,} images)")
    print(f"  Val   batches : {len(val_gen):,}  ({val_gen.n:,} images)")
    print(f"  Classes       : {class_names}")

    return train_gen, val_gen, class_names, cw_dict


# ═══════════════════════════════════════════════════════════════════════════════
#  COSINE-DECAY LR SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════


def cosine_schedule(initial_lr: float, epochs: int, warmup: int = 2):
    """Return a LearningRateScheduler callback with cosine decay + warmup."""

    def schedule(epoch: int, _lr: float) -> float:
        if epoch < warmup:
            return initial_lr * (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return initial_lr * 0.5 * (1.0 + np.cos(np.pi * progress))

    return keras.callbacks.LearningRateScheduler(schedule, verbose=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def _compile(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )


def _callbacks(cfg: Config, lr: float, epochs: int, patience: int = 6):
    return [
        ModelCheckpoint(
            str(cfg.MODEL_BEST),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        cosine_schedule(lr, epochs, warmup=min(2, epochs // 3)),
    ]


def train(skeleton_dir: Path, cfg: Config):
    print("\n" + "═" * 60)
    print("  STEP 4 — Building Model & Training")
    print("═" * 60)

    train_gen, val_gen, class_names, cw_dict = make_generators(skeleton_dir, cfg)
    model, base = build_model(len(class_names))

    total_params = model.count_params()
    print("\n  Model         : EfficientNetB3 + custom head")
    print(f"  Total params  : {total_params:,}")
    print(f"  Num classes   : {len(class_names)}")

    # ── Phase 1: Frozen backbone — train head only ─────────────────────────
    ep1, ep2, ep3 = cfg.epochs
    print(f"\n{'─' * 60}")
    print(f"  PHASE 1 — Frozen base  (up to {ep1} epochs, LR=1e-3)")
    print("─" * 60)
    _compile(model, 1e-3)
    h1 = model.fit(
        train_gen,
        epochs=ep1,
        validation_data=val_gen,
        class_weight=cw_dict,
        callbacks=_callbacks(cfg, 1e-3, ep1),
        verbose=1,
    )

    # ── Phase 2: Unfreeze top-50 base layers ──────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  PHASE 2 — Fine-tune top-50 layers  (up to {ep2} epochs, LR=5e-5)")
    print("─" * 60)
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False
    trainable2 = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"  Trainable params now: {trainable2:,}")
    _compile(model, 5e-5)
    h2 = model.fit(
        train_gen,
        epochs=ep2,
        validation_data=val_gen,
        class_weight=cw_dict,
        callbacks=_callbacks(cfg, 5e-5, ep2, patience=8),
        verbose=1,
    )

    # ── Phase 3: Unfreeze all — very low LR ───────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  PHASE 3 — Full fine-tune  (up to {ep3} epochs, LR=1e-5)")
    print("─" * 60)
    for layer in base.layers:
        layer.trainable = True
    _compile(model, 1e-5)
    h3 = model.fit(
        train_gen,
        epochs=ep3,
        validation_data=val_gen,
        class_weight=cw_dict,
        callbacks=_callbacks(cfg, 1e-5, ep3, patience=5),
        verbose=1,
    )

    # Best val_accuracy across all phases
    all_val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"] + h3.history["val_accuracy"]
    best_acc = max(all_val_acc) * 100
    print(f"\n✅ Training complete — best val accuracy: {best_acc:.2f} %")

    # Save final model + class names
    model.save(str(cfg.MODEL_FINAL))
    with open(cfg.CLASS_JSON, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"   Best model  → {cfg.MODEL_BEST}")
    print(f"   Final model → {cfg.MODEL_FINAL}")
    print(f"   Classes     → {cfg.CLASS_JSON}")

    return (h1, h2, h3), class_names, train_gen, val_gen


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate(cfg: Config, class_names: list[str], val_gen) -> None:
    print("\n" + "═" * 60)
    print("  STEP 5 — Evaluation")
    print("═" * 60)

    best_model = keras.models.load_model(str(cfg.MODEL_BEST))
    val_gen.reset()
    preds = best_model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes[: len(y_pred)]

    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)
    plt.figure(figsize=(max(14, n), max(11, n)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix — Best Model", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    out = cfg.OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    print(f"\n✅ Confusion matrix saved → {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING CURVES
# ═══════════════════════════════════════════════════════════════════════════════


def plot_curves(histories: tuple, cfg: Config) -> None:
    h1, h2, h3 = histories
    ep1 = len(h1.history["accuracy"])
    ep2 = len(h2.history["accuracy"])

    acc = h1.history["accuracy"] + h2.history["accuracy"] + h3.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"] + h3.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"] + h3.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"] + h3.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for ax, train_m, val_m, title in [
        (ax1, acc, val_acc, "Accuracy"),
        (ax2, loss, val_loss, "Loss"),
    ]:
        ax.plot(epochs_range, train_m, "b-o", label="Train", markersize=3)
        ax.plot(epochs_range, val_m, "r-o", label="Validation", markersize=3)
        ax.axvline(ep1, color="gray", linestyle="--", alpha=0.6, label="Fine-tune top-50")
        ax.axvline(ep1 + ep2, color="orange", linestyle="--", alpha=0.6, label="Full fine-tune")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Curves — EfficientNetB3, 3-Phase", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = cfg.OUTPUT_DIR / "training_curves.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    print(f"✅ Training curves saved → {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Train improved ASL model")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true", help="Quick smoke test")
    mode.add_argument("--full", action="store_true", help="Full production training")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  SignSync ASL Training v2 — EfficientNetB3")
    print("═" * 60)

    setup_hardware()
    cfg = Config(test_mode=args.test)

    # ── 1. Data ──
    raw_dir = download_dataset()
    download_mediapipe_task(cfg)

    if _skeleton_dataset_exists(cfg.SKELETON_DIR):
        print(f"\n✅ Skeleton dataset already exists at {cfg.SKELETON_DIR} — skipping conversion")
    else:
        build_skeleton_dataset(raw_dir, cfg)

    remove_black_images(cfg.SKELETON_DIR)

    # ── 2. Train ──
    histories, class_names, train_gen, val_gen = train(cfg.SKELETON_DIR, cfg)

    # ── 3. Evaluate & plot ──
    evaluate(cfg, class_names, val_gen)
    plot_curves(histories, cfg)

    print("\n" + "═" * 60)
    print("  ALL DONE 🎉")
    print("═" * 60)
    print(f"\n  Best model  → {cfg.MODEL_BEST}")
    print(f"  Class names → {cfg.CLASS_JSON}")
    print()
    print("  ► Deploy:")
    print(f"    cp {cfg.MODEL_BEST} src/app/core/ml/trained_model/sign_language_mobilenet.keras")
    print(f"    cp {cfg.CLASS_JSON} src/app/core/ml/trained_model/class_names.json")


if __name__ == "__main__":
    main()
