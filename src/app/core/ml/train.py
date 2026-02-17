"""ASL Sign Language Recognition — LOCAL Training Script Run on your local machine with GPU (NVIDIA + CUDA)

Usage:
    python train_asl_local.py

Requirements:
    pip install tensorflow mediapipe kagglehub opencv-python scikit-learn seaborn matplotlib numpy
"""

import json
import os
import random
import sys
import urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ════════════════════════════════════════════════════
# ⚙️  CONFIGURATION
# ════════════════════════════════════════════════════

TEST_MODE = False  # True = quick test | False = full training

if TEST_MODE:
    BATCH_SIZE = 8
    EPOCHS_PHASE1 = 2
    EPOCHS_PHASE2 = 2
    MAX_CLASSES = 5
    MAX_IMGS_PER_CLS = 50
    print("🧪 TEST MODE — batch=8, epochs=2+2, 5 classes, 50 imgs/class")
else:
    BATCH_SIZE = 32
    EPOCHS_PHASE1 = 15
    EPOCHS_PHASE2 = 20
    MAX_CLASSES = None
    MAX_IMGS_PER_CLS = None
    print("🚀 FULL MODE — batch=32, epochs=15+20, all classes")

# Paths (LOCAL)
BASE_DIR = Path("./asl_training")
SKELETON_DIR = BASE_DIR / "skeleton_dataset"
MODEL_SAVE_PATH = BASE_DIR / "sign_language_model_best.keras"
MODEL_FINAL_PATH = BASE_DIR / "sign_language_model_final.keras"
CLASS_JSON_PATH = BASE_DIR / "class_names.json"
TASK_FILENAME = BASE_DIR / "hand_landmarker.task"
RESULTS_DIR = BASE_DIR / "results"

TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

IMG_SIZE = (224, 224)

# Create directories
BASE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════
# CHECK GPU
# ════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("🔍 Checking GPU...")
print("=" * 50)

os.system("nvidia-smi")

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"\n✅ GPU ready: {gpus}")
else:
    print("\n⚠️  No GPU detected — training will be SLOW on CPU")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != "y":
        sys.exit(0)

# ════════════════════════════════════════════════════
# INSTALL/IMPORT PACKAGES
# ════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("📦 Importing packages...")
print("=" * 50)

try:
    import kagglehub
    import mediapipe as mp
    from sklearn.metrics import classification_report, confusion_matrix

    print(f"✅ mediapipe  {mp.__version__}")
    print(f"✅ opencv     {cv2.__version__}")
    print(f"✅ numpy      {np.__version__}")
    print(f"✅ tensorflow {tf.__version__}")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("\nInstall required packages:")
    print("pip install tensorflow mediapipe kagglehub opencv-python scikit-learn seaborn matplotlib")
    sys.exit(1)

# ════════════════════════════════════════════════════
# SETUP KAGGLE CREDENTIALS
# ════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("🔑 Checking Kaggle credentials...")
print("=" * 50)

kaggle_dir = Path.home() / ".kaggle"
kaggle_json = kaggle_dir / "kaggle.json"

if not kaggle_json.exists():
    print(f"""
❌ Kaggle credentials not found!

To fix:
1. Go to https://www.kaggle.com/settings
2. Click "Create New Token" under API section
3. Download kaggle.json
4. Move it to: {kaggle_dir}

On Windows: C:\\Users\\<YourName>\\.kaggle\\kaggle.json
On Linux/Mac: ~/.kaggle/kaggle.json

Then run this script again.
""")
    sys.exit(1)
else:
    # Set permissions (Linux/Mac)
    try:
        os.chmod(kaggle_json, 0o600)
    except:
        pass
    print(f"✅ Kaggle credentials found: {kaggle_json}")

# ════════════════════════════════════════════════════
# DOWNLOAD DATASET + MEDIAPIPE MODEL
# ════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("⬇️  Downloading ASL Alphabet dataset...")
print("=" * 50)

base_path = kagglehub.dataset_download("grassknoted/asl-alphabet")

RAW_DIR = Path(base_path) / "asl_alphabet_train" / "asl_alphabet_train"
if not RAW_DIR.exists():
    RAW_DIR = Path(base_path) / "asl_alphabet_train"

all_classes = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
print(f"✅ Found {len(all_classes)} classes: {all_classes}")
print(f"   Path: {RAW_DIR}")

# Download MediaPipe hand landmarker
if not TASK_FILENAME.exists():
    print("\n⬇️  Downloading hand_landmarker.task...")
    urllib.request.urlretrieve(TASK_URL, str(TASK_FILENAME))
    size_mb = TASK_FILENAME.stat().st_size / 1e6
    print(f"✅ Downloaded ({size_mb:.1f} MB)")
else:
    print("\n✅ hand_landmarker.task already exists")

# ════════════════════════════════════════════════════
# CORE FUNCTIONS (SKELETON PIPELINE)
# ════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("🔧 Setting up skeleton pipeline...")
print("=" * 50)

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

HAND_CONNECTIONS = [(c.start, c.end) for c in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS]


def to_pixel(val: float, min_val: float, size: float, canvas: int = 224) -> int:
    return max(0, min(canvas - 1, int(((val - min_val) / size) * canvas)))


def build_detector():
    base_options = mp_tasks.BaseOptions(model_asset_path=str(TASK_FILENAME))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
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


print("✅ Core functions defined")

# ════════════════════════════════════════════════════
# GENERATE SKELETON DATASET
# ════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("🦴 Generating skeleton dataset...")
print("=" * 50)


def generate_skeleton_dataset(raw_dir, out_dir, max_classes=None, max_per_class=None):
    classes = sorted([d.name for d in Path(raw_dir).iterdir() if d.is_dir()])
    if max_classes:
        classes = classes[:max_classes]

    total = 0
    for cls in classes:
        imgs = list((Path(raw_dir) / cls).glob("*.jpg"))
        imgs += list((Path(raw_dir) / cls).glob("*.jpeg"))
        imgs += list((Path(raw_dir) / cls).glob("*.png"))
        if max_per_class:
            imgs = imgs[:max_per_class]
        total += len(imgs)

    mode_str = f"{len(classes)} classes × up to {max_per_class or 'all'} imgs"
    print(f"Generating skeletons: {mode_str} = {total} total images")
    print(f"Output: {out_dir}\n")

    detector = build_detector()
    processed = 0
    skipped = 0
    no_hand = 0

    for cls in classes:
        src_dir = Path(raw_dir) / cls
        dst_dir = Path(out_dir) / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        imgs = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.png"))
        if max_per_class:
            imgs = imgs[:max_per_class]

        for img_path in imgs:
            dst_path = dst_dir / img_path.name
            if dst_path.exists():
                skipped += 1
                processed += 1
                continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                processed += 1
                continue

            skeleton, found = image_to_skeleton(bgr, detector)
            if not found:
                no_hand += 1

            cv2.imwrite(str(dst_path), cv2.cvtColor(skeleton, cv2.COLOR_RGB2BGR))
            processed += 1

            if processed % 200 == 0:
                pct = processed / total * 100
                print(f"  [{processed:5}/{total}] {pct:5.1f}%  no_hand={no_hand}  skipped={skipped}")

    new_imgs = processed - skipped
    found_rate = (new_imgs - no_hand) / max(new_imgs, 1) * 100
    print("\n✅ Done!")
    print(f"   New images processed : {new_imgs}")
    print(f"   Already existed      : {skipped}")
    print(f"   Hand detection rate  : {found_rate:.1f}%")
    print(f"   No hand (black img)  : {no_hand}")

    return out_dir


SKELETON_DIR = generate_skeleton_dataset(
    RAW_DIR,
    SKELETON_DIR,
    max_classes=MAX_CLASSES,
    max_per_class=MAX_IMGS_PER_CLS,
)

# ════════════════════════════════════════════════════
# CLEANUP BLACK/EMPTY IMAGES
# ════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("🧹 Cleaning up black/empty images...")
print("=" * 50)


def cleanup_black_images(skeleton_dir):
    removed = 0
    kept = 0

    for class_dir in sorted(Path(skeleton_dir).iterdir()):
        if not class_dir.is_dir():
            continue

        for img_file in class_dir.glob("*.jpg"):
            img = cv2.imread(str(img_file))
            if img is None:
                os.remove(str(img_file))
                removed += 1
                continue

            if np.mean(img) < 5:
                os.remove(str(img_file))
                removed += 1
            else:
                kept += 1

    print("✅ Cleanup complete!")
    print(f"   Kept    : {kept:,} valid skeleton images")
    print(f"   Removed : {removed:,} black/empty images")

    return kept, removed


cleanup_black_images(SKELETON_DIR)

# ════════════════════════════════════════════════════
# CHECK CLASS DISTRIBUTION
# ════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("📊 Checking class distribution...")
print("=" * 50)


def check_class_distribution(skeleton_dir):
    print(f"\n{'Class':<15} {'Count':>8}")
    print("-" * 25)

    total = 0
    class_counts = {}

    for class_dir in sorted(Path(skeleton_dir).iterdir()):
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

        print(f"\nMin: {min_count:,}, Max: {max_count:,}, Balance ratio: {ratio:.2f}")

        if ratio < 0.5:
            print("⚠️  WARNING: Classes imbalanced!")
        else:
            print("✅ Classes reasonably balanced.")

    return class_counts


class_counts = check_class_distribution(SKELETON_DIR)

# ════════════════════════════════════════════════════
# TRAIN MODEL (2-PHASE)
# ════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("🤖 Training model...")
print("=" * 50)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    str(SKELETON_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)
val_gen = datagen.flow_from_directory(
    str(SKELETON_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
)

class_names = list(train_gen.class_indices.keys())
with open(CLASS_JSON_PATH, "w") as f:
    json.dump(class_names, f)

print(f"Classes: {len(class_names)} → {class_names}")
print(f"Train  : {train_gen.samples} images")
print(f"Val    : {val_gen.samples} images")

# Build model
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

print(f"\nTrainable params: {sum(tf.size(v).numpy() for v in model.trainable_variables):,}")

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(str(MODEL_SAVE_PATH), monitor="val_accuracy", save_best_only=True, verbose=1),
]

# Phase 1: Train with frozen base
print(f"\n🚀 PHASE 1: Training with frozen base (max {EPOCHS_PHASE1} epochs)\n")
history1 = model.fit(
    train_gen,
    epochs=EPOCHS_PHASE1,
    validation_data=val_gen,
    callbacks=callbacks,
)

# Phase 2: Fine-tune
print(f"\n🚀 PHASE 2: Fine-tuning last 30 layers (max {EPOCHS_PHASE2} epochs)\n")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

history2 = model.fit(
    train_gen,
    epochs=EPOCHS_PHASE2,
    validation_data=val_gen,
    callbacks=callbacks,
)

model.save(str(MODEL_FINAL_PATH))

best = max(history2.history["val_accuracy"]) * 100
print(f"\n✅ Training complete — best val accuracy: {best:.2f}%")
print(f"   Best model : {MODEL_SAVE_PATH}")
print(f"   Final model: {MODEL_FINAL_PATH}")

# ════════════════════════════════════════════════════
# EVALUATE WITH CONFUSION MATRIX
# ════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("📊 Evaluating model...")
print("=" * 50)

best_model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))

val_gen.reset()
predictions = best_model.predict(val_gen, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_gen.classes[: len(y_pred)]

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "confusion_matrix.png"), dpi=120, bbox_inches="tight")
plt.show()
print(f"\nSaved: {RESULTS_DIR / 'confusion_matrix.png'}")

# ════════════════════════════════════════════════════
# PLOT TRAINING CURVES
# ════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("📈 Plotting training curves...")
print("=" * 50)

combined_acc = history1.history["accuracy"] + history2.history["accuracy"]
combined_val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
combined_loss = history1.history["loss"] + history2.history["loss"]
combined_val_loss = history1.history["val_loss"] + history2.history["val_loss"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_total = range(1, len(combined_acc) + 1)
phase1_end = len(history1.history["accuracy"])

ax1.plot(epochs_total, combined_acc, "b-o", label="Train", markersize=4)
ax1.plot(epochs_total, combined_val_acc, "r-o", label="Validation", markersize=4)
ax1.axvline(phase1_end, color="gray", linestyle="--", label="Fine-tune starts", alpha=0.7)
ax1.set_title("Accuracy", fontsize=13, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_total, combined_loss, "b-o", label="Train", markersize=4)
ax2.plot(epochs_total, combined_val_loss, "r-o", label="Validation", markersize=4)
ax2.axvline(phase1_end, color="gray", linestyle="--", label="Fine-tune starts", alpha=0.7)
ax2.set_title("Loss", fontsize=13, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

mode_label = "TEST MODE" if TEST_MODE else "FULL TRAINING"
fig.suptitle(f"Training Curves [{mode_label}] — 2 Phases", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "training_curves.png"), dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved: {RESULTS_DIR / 'training_curves.png'}")

# ════════════════════════════════════════════════════
# TEST ON RANDOM SAMPLES
# ════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("🧪 Testing on random samples...")
print("=" * 50)

CONFIDENCE_THRESHOLD = 0.5


def predict_single(model, class_names, skeleton_float32):
    probs = model.predict(skeleton_float32[np.newaxis], verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    label = class_names[top_idx] if top_conf >= CONFIDENCE_THRESHOLD else "Uncertain"
    return label, top_conf


def test_random_samples(skeleton_dir, model, class_names, num_samples=12):
    classes = sorted([d.name for d in Path(skeleton_dir).iterdir() if d.is_dir()])
    chosen = random.sample(classes, min(num_samples, len(classes)))

    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 4))
    axes = axes.flatten()

    correct = 0
    for i, cls in enumerate(chosen):
        imgs = list((Path(skeleton_dir) / cls).glob("*.jpg"))
        img_path = random.choice(imgs)
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        skel = cv2.resize(rgb, (224, 224)).astype("float32") / 255.0

        pred, conf = predict_single(model, class_names, skel)
        is_correct = pred == cls
        if is_correct:
            correct += 1

        axes[i].imshow(rgb)
        color = "green" if is_correct else "red"
        axes[i].set_title(f"True: {cls}\nPred: {pred} ({conf * 100:.0f}%)", fontsize=11, color=color, fontweight="bold")
        axes[i].axis("off")

    for j in range(len(chosen), len(axes)):
        axes[j].axis("off")

    accuracy = correct / len(chosen) * 100
    fig.suptitle(
        f"Sample Predictions — {correct}/{len(chosen)} correct ({accuracy:.0f}%)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "sample_predictions.png"), dpi=120, bbox_inches="tight")
    plt.show()
    print(f"\nSample accuracy: {correct}/{len(chosen)} ({accuracy:.0f}%)")


test_random_samples(SKELETON_DIR, best_model, class_names, num_samples=12)

# ════════════════════════════════════════════════════
# DONE!
# ════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("🎉 TRAINING COMPLETE!")
print("=" * 50)
print(f"""
📁 Output files:
   {MODEL_SAVE_PATH}
   {MODEL_FINAL_PATH}
   {CLASS_JSON_PATH}
   {RESULTS_DIR / "confusion_matrix.png"}
   {RESULTS_DIR / "training_curves.png"}
   {RESULTS_DIR / "sample_predictions.png"}

🚀 Next steps:
   1. Copy model files to your FastAPI backend:
      - {MODEL_SAVE_PATH.name}
      - {CLASS_JSON_PATH.name}

   2. Update FastAPI endpoint to load this model

   3. Test with Angular frontend
""")
