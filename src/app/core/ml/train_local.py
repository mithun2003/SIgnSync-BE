import json
import os
import urllib.request

import cv2
import kagglehub
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 0. GPU CONFIGURATION ---
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Enabled: {gpus}")
    except RuntimeError as e:
        print(e)

# --- CONFIGURATION ---
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 5
TASK_FILENAME = "hand_landmarker.task"

# --- HAND CONNECTIONS ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# --- 1. DOWNLOAD DATASET ---
def download_dataset():
    print("⬇️ Downloading Dataset via KaggleHub...")
    base_path = kagglehub.dataset_download("grassknoted/asl-alphabet")

    train_dir = os.path.join(base_path, "asl_alphabet_train", "asl_alphabet_train")
    if not os.path.exists(train_dir):
        train_dir = os.path.join(base_path, "asl_alphabet_train")

    print(f"✅ Data located at: {train_dir}")
    return train_dir


# --- 2. SETUP MEDIAPIPE ---
def setup_mediapipe():
    if not os.path.exists(TASK_FILENAME):
        print(f"⬇️ Downloading {TASK_FILENAME}...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, TASK_FILENAME)
        print("✅ Download Complete.")

    base_options = python.BaseOptions(model_asset_path=TASK_FILENAME)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.3)
    return vision.HandLandmarker.create_from_options(options)


detector = None


# --- 3. SKELETON PREPROCESSING ---
def smart_preprocess(img):
    global detector
    skeleton_img = np.zeros((224, 224, 3), dtype=np.uint8)

    if detector is None:
        return skeleton_img.astype("float32") / 255.0

    try:
        img_uint8 = img.astype(np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_uint8)
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]

            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            pad_x = (x_max - x_min) * 0.1
            pad_y = (y_max - y_min) * 0.1
            x_min -= pad_x
            x_max += pad_x
            y_min -= pad_y
            y_max += pad_y

            width = x_max - x_min
            height = y_max - y_min
            if width <= 0:
                width = 0.1
            if height <= 0:
                height = 0.1

            def to_pixel(val, min_val, size):
                return int(((val - min_val) / size) * 224)

            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = landmarks[start_idx]
                p2 = landmarks[end_idx]
                pt1 = (to_pixel(p1.x, x_min, width), to_pixel(p1.y, y_min, height))
                pt2 = (to_pixel(p2.x, x_min, width), to_pixel(p2.y, y_min, height))
                cv2.line(skeleton_img, pt1, pt2, (255, 255, 255), 2)

            for lm in landmarks:
                px = to_pixel(lm.x, x_min, width)
                py = to_pixel(lm.y, y_min, height)
                cv2.circle(skeleton_img, (px, py), 3, (255, 255, 255), -1)

        return skeleton_img.astype("float32") / 255.0

    except Exception:
        return np.zeros((224, 224, 3), dtype="float32")


# --- MAIN BLOCK ---
if __name__ == "__main__":
    TRAIN_DIR = download_dataset()
    detector = setup_mediapipe()

    train_datagen = ImageDataGenerator(
        preprocessing_function=smart_preprocess,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    print("\n⏳ Preparing Data Generators...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(224, 224), batch_size=BATCH_SIZE, class_mode="categorical", subset="validation"
    )

    class_names = list(train_generator.class_indices.keys())
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)

    # Build Model
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(len(class_names), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

    # Train
    print("\n🚀 Training Started...")
    history = model.fit(
        train_generator, epochs=EPOCHS, validation_data=validation_generator, workers=1, use_multiprocessing=False
    )

    model.save("sign_language_mobilenet_skeleton.keras")
    print("\n✅ Model Saved.")
