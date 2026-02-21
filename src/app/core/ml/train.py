"""ASL Sign Language Recognition — Local Training Script ══════════════════════════════════════════════════════
Converted from Colab notebook to run on your local machine (Windows/Linux).

IMPROVEMENTS:
- Removes black/empty skeleton images before training
- Checks class balance
- Two-phase training (frozen base + fine-tuning)
- Confusion matrix + classification report
- Works on both GPU and CPU

REQUIREMENTS:
- Python 3.8+
- CUDA 11.8 + cuDNN 8.6 (for GPU) OR just CPU
- See requirements.txt for packages

USAGE:
    python train_local.py --test          # Quick test (5 min)
    python train_local.py --full          # Full training (30-45 min)
"""

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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ═══════════════════════════════════════════════════════════════════════════
#  GPU SETUP
# ═══════════════════════════════════════════════════════════════════════════
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU enabled: {gpus}")
            return True
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    else:
        print("ℹ️  No GPU found — running on CPU (will be slower)")
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    def __init__(self, test_mode=False):
        self.test_mode = test_mode

        if test_mode:
            self.batch_size = 8
            self.epochs_phase1 = 2
            self.epochs_phase2 = 2
            self.max_classes = 5
            self.max_imgs_per_cls = 50
            print("🧪 TEST MODE — batch=8, epochs=2+2, 5 classes, 50 imgs/class")
            print("   Expected time: ~5-10 min on GPU, ~20-30 min on CPU")
        else:
            self.batch_size = 32
            self.epochs_phase1 = 15
            self.epochs_phase2 = 20
            self.max_classes = None
            self.max_imgs_per_cls = None
            print("🚀 FULL MODE — batch=32, epochs=15+20, all 29 classes")
            print("   Expected time: ~30-45 min on GPU, ~2-4 hours on CPU")

        # Fixed paths
        self.img_size = (224, 224)
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        self.skeleton_dir = self.output_dir / "skeleton_dataset"
        self.model_best_path = self.output_dir / "sign_language_model_best.keras"
        self.model_final_path = self.output_dir / "sign_language_model_final.keras"
        self.class_json_path = self.output_dir / "class_names.json"
        self.task_filename = self.output_dir / "hand_landmarker.task"
        self.task_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
HAND_CONNECTIONS = None  # Will be set after MediaPipe loads


def to_pixel(val: float, min_val: float, size: float, canvas: int = 224) -> int:
    return max(0, min(canvas - 1, int(((val - min_val) / size) * canvas)))


def build_detector(task_path: Path):
    global HAND_CONNECTIONS
    base_options = mp_tasks.BaseOptions(model_asset_path=str(task_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    HAND_CONNECTIONS = [(c.start, c.end) for c in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS]
    return mp_vision.HandLandmarker.create_from_options(options)


def image_to_skeleton(img_bgr: np.ndarray, detector) -> tuple:
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    img_uint8 = np.ascontiguousarray(img_rgb)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_uint8)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return canvas, False

    lms = result.hand_landmarks[0]
    xs = [lm.x for lm in lms]
    ys = [lm.y for lm in lms]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    pad_x = (x1 - x0) * 0.15
    pad_y = (y1 - y0) * 0.15
    x0 -= pad_x
    x1 += pad_x
    y0 -= pad_y
    y1 += pad_y
    w = max(x1 - x0, 0.01)
    h = max(y1 - y0, 0.01)

    for s, e in HAND_CONNECTIONS:
        p1, p2 = lms[s], lms[e]
        cv2.line(
            canvas,
            (to_pixel(p1.x, x0, w), to_pixel(p1.y, y0, h)),
            (to_pixel(p2.x, x0, w), to_pixel(p2.y, y0, h)),
            (255, 255, 255),
            2,
        )

    for lm in lms:
        cv2.circle(canvas, (to_pixel(lm.x, x0, w), to_pixel(lm.y, y0, h)), 4, (220, 220, 220), -1)

    return canvas, True


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════
def download_dataset():
    print("\n" + "═" * 60)
    print("  STEP 1 — Downloading ASL Alphabet Dataset")
    print("═" * 60)

    base_path = kagglehub.dataset_download("grassknoted/asl-alphabet")

    train_dir = os.path.join(base_path, "asl_alphabet_train", "asl_alphabet_train")
    if not os.path.exists(train_dir):
        train_dir = os.path.join(base_path, "asl_alphabet_train")

    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"✅ {len(classes)} classes found: {classes}")
    print(f"   Path: {train_dir}")
    return train_dir


def download_mediapipe_model(task_path: Path, task_url: str):
    if task_path.exists():
        print(f"✅ {task_path.name} already exists")
        return

    print(f"\n⬇️  Downloading {task_path.name}...")
    urllib.request.urlretrieve(task_url, str(task_path))
    size_mb = task_path.stat().st_size / 1e6
    print(f"✅ Downloaded ({size_mb:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════════
#  SKELETON GENERATION
# ═══════════════════════════════════════════════════════════════════════════
def generate_skeleton_dataset(raw_dir: str, out_dir: Path, config: Config, task_path: Path):
    print("\n" + "═" * 60)
    print("  STEP 2 — Pre-generating Skeleton Dataset")
    print("═" * 60)

    classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    if config.max_classes:
        classes = classes[: config.max_classes]

    total = 0
    for cls in classes:
        imgs = [f for f in os.listdir(os.path.join(raw_dir, cls)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if config.max_imgs_per_cls:
            imgs = imgs[: config.max_imgs_per_cls]
        total += len(imgs)

    print(f"  Classes     : {len(classes)}")
    print(f"  Total images: {total:,}")
    print(f"  Output dir  : {out_dir}\n")

    detector = build_detector(task_path)
    processed = 0
    skipped = 0
    no_hand = 0

    for cls in classes:
        src_dir = os.path.join(raw_dir, cls)
        dst_dir = out_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        imgs = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if config.max_imgs_per_cls:
            imgs = imgs[: config.max_imgs_per_cls]

        for fname in imgs:
            dst_path = dst_dir / fname
            if dst_path.exists():
                skipped += 1
                processed += 1
                continue

            bgr = cv2.imread(os.path.join(src_dir, fname))
            if bgr is None:
                processed += 1
                continue

            skeleton, found = image_to_skeleton(bgr, detector)
            if not found:
                no_hand += 1

            cv2.imwrite(str(dst_path), cv2.cvtColor(skeleton, cv2.COLOR_RGB2BGR))
            processed += 1

            if processed % 500 == 0:
                pct = processed / total * 100
                print(f"  [{processed:6}/{total}] {pct:5.1f}%  no_hand={no_hand}  skipped={skipped}")

    new_imgs = processed - skipped
    found_rate = (new_imgs - no_hand) / max(new_imgs, 1) * 100
    print("\n✅ Done!")
    print(f"   New processed : {new_imgs:,}")
    print(f"   Skipped       : {skipped:,}")
    print(f"   Hand found    : {found_rate:.1f}%")
    print(f"   No hand       : {no_hand:,}")

    return out_dir


# ═══════════════════════════════════════════════════════════════════════════
#  CLEANUP BLACK IMAGES
# ═══════════════════════════════════════════════════════════════════════════
def cleanup_black_images(skeleton_dir: Path):
    print("\n" + "═" * 60)
    print("  STEP 3 — Removing Black/Empty Images")
    print("═" * 60)

    removed = 0
    kept = 0

    for class_dir in sorted(skeleton_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for img_file in class_dir.glob("*.jpg"):
            img = cv2.imread(str(img_file))
            if img is None:
                img_file.unlink()
                removed += 1
                continue

            if np.mean(img) < 5:
                img_file.unlink()
                removed += 1
            else:
                kept += 1

    print(f"  Kept    : {kept:,} valid skeleton images")
    print(f"  Removed : {removed:,} black/empty images")
    return kept, removed


# ═══════════════════════════════════════════════════════════════════════════
#  CLASS DISTRIBUTION CHECK
# ═══════════════════════════════════════════════════════════════════════════
def check_class_distribution(skeleton_dir: Path):
    print("\n" + "═" * 60)
    print("  STEP 4 — Class Distribution")
    print("═" * 60)

    print(f"\n{'Class':<15} {'Count':>8}")
    print("-" * 25)

    total = 0
    class_counts = {}

    for class_dir in sorted(skeleton_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        count = len(list(class_dir.glob("*.jpg")))
        class_counts[class_dir.name] = count
        total += count

        status = "⚠️ " if count < 500 else "✅"
        print(f"{status} {class_dir.name:<15} {count:>8,}")

    print("-" * 25)
    print(f"{'Total':<15} {total:>8,}")

    if class_counts:
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        ratio = min_count / max_count if max_count > 0 else 0

        print(f"\nMin: {min_count:,}, Max: {max_count:,}, Balance: {ratio:.2f}")

        if ratio < 0.5:
            print("⚠️  WARNING: Imbalanced! Model may favor larger classes.")
        else:
            print("✅ Classes reasonably balanced.")

    return class_counts


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════
def train_model(skeleton_dir: Path, config: Config):
    print("\n" + "═" * 60)
    print("  STEP 5 — Training Model (2-Phase)")
    print("═" * 60)

    # Data generators
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )

    train_gen = datagen.flow_from_directory(
        str(skeleton_dir),
        target_size=config.img_size,
        batch_size=config.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        str(skeleton_dir),
        target_size=config.img_size,
        batch_size=config.batch_size,
        class_mode="categorical",
        subset="validation",
    )

    class_names = list(train_gen.class_indices.keys())
    with open(config.class_json_path, "w") as f:
        json.dump(class_names, f)

    print(f"\nClasses: {len(class_names)} → {class_names}")
    print(f"Train  : {train_gen.samples:,} images")
    print(f"Val    : {val_gen.samples:,} images")

    # Build model
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(len(class_names), activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"\nTrainable params: {trainable:,}")

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            str(config.model_best_path), monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    # Phase 1: Frozen base
    print(f"\n🚀 PHASE 1: Frozen base (max {config.epochs_phase1} epochs)\n")
    history1 = model.fit(
        train_gen,
        epochs=config.epochs_phase1,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # Phase 2: Fine-tune
    print(f"\n🚀 PHASE 2: Fine-tuning (max {config.epochs_phase2} epochs)\n")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

    history2 = model.fit(
        train_gen,
        epochs=config.epochs_phase2,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    model.save(str(config.model_final_path))

    best = max(history2.history["val_accuracy"]) * 100
    print(f"\n✅ Training complete — best val accuracy: {best:.2f}%")
    print(f"   Best  : {config.model_best_path}")
    print(f"   Final : {config.model_final_path}")

    return history1, history2, class_names


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model(skeleton_dir: Path, config: Config, class_names: list):
    print("\n" + "═" * 60)
    print("  STEP 6 — Evaluation")
    print("═" * 60)

    model = tf.keras.models.load_model(str(config.model_best_path))

    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
    val_gen = datagen.flow_from_directory(
        str(skeleton_dir),
        target_size=config.img_size,
        batch_size=config.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    predictions = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes[: len(y_pred)]

    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(12, len(class_names)), max(10, len(class_names))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    cm_path = config.output_dir / "confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=120, bbox_inches="tight")
    print(f"\n✅ Saved: {cm_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT TRAINING CURVES
# ═══════════════════════════════════════════════════════════════════════════
def plot_training_curves(history1, history2, config: Config):
    combined_acc = history1.history["accuracy"] + history2.history["accuracy"]
    combined_val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    combined_loss = history1.history["loss"] + history2.history["loss"]
    combined_val_loss = history1.history["val_loss"] + history2.history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_total = range(1, len(combined_acc) + 1)
    phase1_end = len(history1.history["accuracy"])

    ax1.plot(epochs_total, combined_acc, "b-o", label="Train", markersize=4)
    ax1.plot(epochs_total, combined_val_acc, "r-o", label="Validation", markersize=4)
    ax1.axvline(phase1_end, color="gray", linestyle="--", label="Fine-tune", alpha=0.7)
    ax1.set_title("Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_total, combined_loss, "b-o", label="Train", markersize=4)
    ax2.plot(epochs_total, combined_val_loss, "r-o", label="Validation", markersize=4)
    ax2.axvline(phase1_end, color="gray", linestyle="--", label="Fine-tune", alpha=0.7)
    ax2.set_title("Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    mode = "TEST MODE" if config.test_mode else "FULL TRAINING"
    fig.suptitle(f"Training Curves [{mode}] — 2 Phases", fontsize=14, fontweight="bold")
    plt.tight_layout()

    curves_path = config.output_dir / "training_curves.png"
    plt.savefig(str(curves_path), dpi=120, bbox_inches="tight")
    print(f"✅ Saved: {curves_path}")
    plt.close()


def skeleton_dataset_exists(skeleton_dir: Path):
    if not skeleton_dir.exists():
        return False

    # Check if at least one class folder has images
    for class_dir in skeleton_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))
            if len(images) > 50:  # enough to confirm dataset exists
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train ASL sign language model")
    parser.add_argument("--test", action="store_true", help="Run in TEST mode")
    parser.add_argument("--full", action="store_true", help="Run in FULL mode")
    args = parser.parse_args()

    if not args.test and not args.full:
        print("ERROR: Must specify either --test or --full")
        print("Usage: python train_local.py --test")
        print("   or: python train_local.py --full")
        return

    config = Config(test_mode=args.test)
    has_gpu = setup_gpu()

    # Download dataset
    raw_dir = download_dataset()

    # Download MediaPipe model
    download_mediapipe_model(config.task_filename, config.task_url)

    # Generate skeletons
    # generate_skeleton_dataset(raw_dir, config.skeleton_dir, config, config.task_filename)

    # Cleanup black images
    # cleanup_black_images(config.skeleton_dir)
    if skeleton_dataset_exists(config.skeleton_dir):
        print("\n✅ Skeleton dataset already exists — skipping generation.")
    else:
        generate_skeleton_dataset(raw_dir, config.skeleton_dir, config, config.task_filename)
    cleanup_black_images(config.skeleton_dir)

    # Check distribution
    check_class_distribution(config.skeleton_dir)

    # Train
    history1, history2, class_names = train_model(config.skeleton_dir, config)

    # Evaluate
    evaluate_model(config.skeleton_dir, config, class_names)

    # Plot curves
    plot_training_curves(history1, history2, config)

    print("\n" + "═" * 60)
    print("  ALL DONE 🎉")
    print("═" * 60)
    print(f"  Output directory: {config.output_dir}")
    print(f"  Best model      : {config.model_best_path}")
    print(f"  Final model     : {config.model_final_path}")
    print(f"  Class names     : {config.class_json_path}")
    print("\n  Upload the .keras file and class_names.json to your FastAPI backend!")


if __name__ == "__main__":
    main()
