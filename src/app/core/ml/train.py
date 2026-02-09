# # 1. Install Library
# !pip install -q kagglehub

# import kagglehub
# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import load_model
# from sklearn.metrics import classification_report, confusion_matrix

# # --- 1. DOWNLOAD DATA ---
# print("⬇️ Downloading Dataset...")
# base_path = kagglehub.dataset_download("grassknoted/asl-alphabet")

# # Define Paths based on your screenshot structure
# TRAIN_DIR = os.path.join(base_path, "asl_alphabet_train", "asl_alphabet_train")
# TEST_DIR = os.path.join(base_path, "asl_alphabet_test", "asl_alphabet_test")

# print(f"✅ Data loaded from: {base_path}")

# # --- 2. DATA GENERATORS (Split TRAIN folder 80/20) ---
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32

# # We use the TRAIN folder for both training and validation because it has the right structure
# data_gen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2, # 20% for checking accuracy during training
#     horizontal_flip=True
# )

# print("\nProcessing Training Data...")
# train_generator = data_gen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'
# )

# print("Processing Validation Data...")
# validation_generator = data_gen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )

# # Save class names for later
# class_names = list(train_generator.class_indices.keys())

# # --- 3. TRAIN MODEL ---
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# base_model.trainable = False

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.2)(x)
# predictions = Dense(len(class_names), activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)
# model.compile(optimizer=Adam(learning_rate=0.0001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# print("\n🚀 Starting Training (This takes time)...")
# # Train for just 3 epochs for demonstration (increase to 5-10 for best results)
# history = model.fit(
#     train_generator,
#     epochs=3,
#     validation_data=validation_generator
# )

# # --- 4. USE THE TEST FOLDER (Manual Evaluation) ---
# # Since the 'test' folder doesn't have subfolders, we loop manually.
# print("\n" + "="*40)
# print("🧪 TESTING ON 'asl_alphabet_test' FOLDER")
# print("="*40)

# test_files = os.listdir(TEST_DIR)
# correct_count = 0
# total_count = 0

# plt.figure(figsize=(12, 6))

# for i, file in enumerate(test_files):
#     if not file.endswith((".jpg", ".png", ".jpeg")):
#         continue

#     # 1. Get True Label from Filename (e.g. "A_test.jpg" -> "A")
#     true_label = file.split('_')[0]

#     # Skip if label not in our classes (e.g. some extra files)
#     if true_label not in class_names:
#         continue

#     # 2. Prepare Image
#     img_path = os.path.join(TEST_DIR, file)
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_resized = cv2.resize(img, IMG_SIZE)
#     img_array = np.expand_dims(img_resized, axis=0).astype('float32') / 255.0

#     # 3. Predict
#     preds = model.predict(img_array, verbose=0)
#     pred_idx = np.argmax(preds)
#     pred_label = class_names[pred_idx]
#     confidence = np.max(preds) * 100

#     # 4. Check Result
#     is_correct = (pred_label == true_label)
#     if is_correct:
#         correct_count += 1
#     total_count += 1

#     # Show first 5 images
#     if i < 5:
#         plt.subplot(1, 5, i+1)
#         plt.imshow(img_resized)
#         color = 'green' if is_correct else 'red'
#         plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
#         plt.axis('off')

# plt.tight_layout()
# plt.show()

# print(f"\n🏆 Final Test Accuracy: {correct_count}/{total_count} ({correct_count/total_count*100:.2f}%)")

# # Save the model for your backend
# model.save("sign_language_mobilenet.keras")
# print("✅ Model Saved.")

# ==========================================
#  ASL ALPHABET - ROBUST GRAYSCALE TRAINING
# ==========================================
# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import kagglehub
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import classification_report

# # --- 1. DOWNLOAD DATA ---
# print("⬇️ Downloading Dataset...")
# base_path = kagglehub.dataset_download("grassknoted/asl-alphabet")

# # Fix: Use the nested structure often found in this dataset
# TRAIN_DIR = os.path.join(base_path, "asl_alphabet_train", "asl_alphabet_train")
# TEST_DIR = os.path.join(base_path, "asl_alphabet_test", "asl_alphabet_test")

# print(f"✅ Data Path: {TRAIN_DIR}")

# # --- 2. GRAYSCALE PREPROCESSING (The Magic Sauce) ---
# def preprocess_grayscale(img):
#     """
#     Converts RGB image to Grayscale, then stacks it back to 3 channels.
#     This creates a B&W image that MobileNetV2 (which expects 3 channels) accepts.
#     """
#     # 1. Convert to Gray
#     # ImageDataGenerator passes images as RGB (0-255)
#     img = tf.image.rgb_to_grayscale(img)

#     # 2. Stack back to 3 channels (R=Gray, G=Gray, B=Gray)
#     img = tf.image.grayscale_to_rgb(img)

#     # 3. Normalize to 0-1 range
#     return img / 255.0

# # --- 3. DATA GENERATORS ---
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32

# # Note: We do NOT use rescale=1./255 here because our function does it manually
# train_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_grayscale, # Apply Grayscale logic
#     validation_split=0.2,                        # Keep 20% for checking accuracy
#     rotation_range=20,
#     zoom_range=0.2,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True
# )

# print("\nProcessing Training Data...")
# train_generator = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training',
#     shuffle=True
# )

# print("Processing Validation Data...")
# validation_generator = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )

# # Save Class Names
# class_names = list(train_generator.class_indices.keys())
# with open("class_names.txt", "w") as f:
#     f.write("\n".join(class_names))

# # --- 4. MODEL SETUP ---
# # MobileNetV2 expects 3 channels. Our grayscale function provides 3 (pseudo-gray) channels.
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# base_model.trainable = False

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.3)(x)
# predictions = Dense(len(class_names), activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)
# model.compile(optimizer=Adam(learning_rate=0.0001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # --- 5. TRAIN ---
# print("\n🚀 Training Started...")
# history = model.fit(
#     train_generator,
#     epochs=5,  # 5 Epochs is usually enough for Transfer Learning
#     validation_data=validation_generator
# )

# # --- 6. TEST ON OFFICIAL TEST SET ---
# print("\n🧪 Testing on official test set...")
# test_files = os.listdir(TEST_DIR)
# correct = 0
# total = 0

# for file in test_files:
#     if not file.endswith((".jpg", ".png")):
#         continue

#     true_label = file.split('_')[0]
#     if true_label not in class_names:
#         continue

#     # Load & Preprocess Manually
#     img_path = os.path.join(TEST_DIR, file)
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, IMG_SIZE)

#     # Apply EXACTLY the same processing as training
#     img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
#     img_processed = preprocess_grayscale(img_tensor) # Gray + Stack + Normalize
#     img_batch = np.expand_dims(img_processed, axis=0)

#     # Predict
#     preds = model.predict(img_batch, verbose=0)
#     pred_label = class_names[np.argmax(preds)]

#     if pred_label == true_label:
#         correct += 1
#     total += 1

# print(f"🏆 Test Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")

# # Save
# model.save("sign_language_mobilenet_grayscale.keras")
# print("✅ Model Saved!")


# ==========================================
#  ASL ALPHABET - ROBUST GRAYSCALE TRAINING WITH CROP
# ==========================================
# 1. Install Dependencies
# !pip install -q kagglehub mediapipe

import json
import os

import cv2
import kagglehub
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 1. DOWNLOAD DATA ---
print("⬇️ Downloading Dataset...")
base_path = kagglehub.dataset_download("grassknoted/asl-alphabet")

TRAIN_DIR = os.path.join(base_path, "asl_alphabet_train", "asl_alphabet_train")
print(f"✅ Data Path: {TRAIN_DIR}")

# --- 2. SETUP MEDIAPIPE (Tasks API) ---
TASK_FILE = "hand_landmarker.task"

# Download the model file if it doesn't exist
if not os.path.exists(TASK_FILE):
    print("⬇️ Downloading MediaPipe Hand Tracker...")
    # !wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    print("✅ Download Complete.")

# Create the Detector (Global Scope)
base_options = python.BaseOptions(model_asset_path=TASK_FILE)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.3)
detector = vision.HandLandmarker.create_from_options(options)


# --- 3. PREPROCESSING FUNCTION (Crop -> Gray -> Stack) ---
def smart_preprocess(img):
    """
    1. Detects Hand & Crops using Modern Tasks API.
    2. Resizes to 224x224.
    3. Converts to Grayscale.
    4. Normalizes (0-1).
    """
    # A. Convert to uint8 for MediaPipe
    img_uint8 = img.astype(np.uint8)

    # B. Convert to MediaPipe Image Object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_uint8)

    # C. Detect
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        # Hand Found! Calculate Bounding Box
        landmarks = detection_result.hand_landmarks[0]  # Get first hand
        h, w, _ = img_uint8.shape

        x_min = min([lm.x for lm in landmarks])
        y_min = min([lm.y for lm in landmarks])
        x_max = max([lm.x for lm in landmarks])
        y_max = max([lm.y for lm in landmarks])

        # Add Padding (10%)
        pad = 0.1
        x_min = max(0, int((x_min - pad) * w))
        y_min = max(0, int((y_min - pad) * h))
        x_max = min(w, int((x_max + pad) * w))
        y_max = min(h, int((y_max + pad) * h))

        # Crop the hand
        hand_crop = img_uint8[y_min:y_max, x_min:x_max]

        # Safety check: If crop is valid, use it
        if hand_crop.size != 0:
            img_uint8 = hand_crop

    # D. Resize to Target Size (MobileNet needs 224x224)
    img_resized = cv2.resize(img_uint8, (224, 224))

    # E. Grayscale Logic (Gray -> RGB Stack)
    # This removes color bias (skin tone/lighting)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    img_stack = np.stack((img_gray,) * 3, axis=-1)

    # F. Normalize 0-1
    return img_stack.astype("float32") / 255.0


# --- 4. DATA GENERATORS ---
BATCH_SIZE = 32

# We do NOT set rescale=1./255 here because our function handles it
train_datagen = ImageDataGenerator(
    preprocessing_function=smart_preprocess,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

print("\nProcessing Training Data (This might be slow due to cropping)...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(250, 250),  # Load slightly larger to allow clean cropping
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(250, 250), batch_size=BATCH_SIZE, class_mode="categorical", subset="validation"
)

# Save Class Names
class_names = list(train_generator.class_indices.keys())
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# --- 5. MODEL SETUP ---
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# --- 6. TRAIN ---
print("\n🚀 Training Started...")
history = model.fit(train_generator, epochs=5, validation_data=validation_generator)

# --- 7. SAVE MODEL ---
print("\n💾 Saving Model...")
model.save("sign_language_mobilenet.keras")
print("\n✅ DONE! Download 'sign_language_mobilenet.keras' and 'class_names.json'.")
