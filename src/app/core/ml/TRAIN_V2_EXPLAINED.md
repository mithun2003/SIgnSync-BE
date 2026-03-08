# `train_v2.py` — Explained for the Team

> **Purpose:** Train an improved, production-ready ASL (American Sign Language) hand-sign recognition model using EfficientNetB3 and a MediaPipe-powered skeleton pipeline.

______________________________________________________________________

## Why This Script Exists

The original model had several accuracy problems that this script fixes:

| Problem (old model)                        | Fix in `train_v2.py`                            |
| ------------------------------------------ | ----------------------------------------------- |
| Only 5 epochs — model never converged      | 3-phase training: 15 + 20 + 10 epochs           |
| Wrong input scale (`/255`) for MobileNetV2 | Preprocessing embedded *inside* the model       |
| No fine-tuning — only the head was trained | Progressive unfreezing of backbone layers       |
| Minimal augmentation                       | Rich augmentation (rotation, zoom, shear, flip) |
| No class-weight balancing                  | `compute_class_weight("balanced")`              |

**Expected result:** ≥ 95% validation accuracy on the full 87k-image dataset.

______________________________________________________________________

## How to Run

```bash
# Quick smoke test — 5 classes, 10 imgs/class, ~5 min
python train_v2.py --test

# Full production training — all 29 classes, ~45 min GPU / 3–4 h CPU
python train_v2.py --full
```

**Requirements:**

```
pip install tensorflow>=2.13 kagglehub opencv-python mediapipe scikit-learn matplotlib seaborn
```

You also need a Kaggle API token saved at `~/.kaggle/kaggle.json`.

______________________________________________________________________

## Overall Flow

```
main()
  │
  ├─ 1. setup_hardware()           — GPU detection + mixed precision
  ├─ 2. download_dataset()         — Kaggle ASL Alphabet (87k images, 29 classes)
  ├─ 3. download_mediapipe_task()  — hand_landmarker.task model file
  ├─ 4. build_skeleton_dataset()   — convert raw photos → skeleton images
  ├─ 5. remove_black_images()      — clean up failed detections
  ├─ 6. train()                    — 3-phase model training
  ├─ 7. evaluate()                 — confusion matrix + classification report
  └─ 8. plot_curves()              — accuracy/loss training curves
```

______________________________________________________________________

## Module-by-Module Breakdown

### `setup_hardware()`

Detects available GPUs and enables:

- **Memory growth** — prevents TensorFlow from grabbing all GPU VRAM at once.
- **Mixed precision (`float16`)** — doubles throughput on RTX 20xx+ GPUs with no accuracy loss.

______________________________________________________________________

### `Config` class

Centralises all hyperparameters. Two modes:

| Setting              | `--test`  | `--full`       |
| -------------------- | --------- | -------------- |
| `batch_size`         | 8         | 32             |
| `epochs`             | (3, 3, 2) | (15, 20, 10)   |
| `max_classes`        | 5         | All 29         |
| `max_imgs_per_class` | 10        | All (~3k each) |

Also defines all output paths (`output_v2/`).

______________________________________________________________________

### Skeleton Generation (MediaPipe)

#### `_build_detector(task_path)`

Creates a **MediaPipe HandLandmarker** configured for single-image mode with relaxed confidence thresholds (0.3) to maximise coverage on dataset images.

#### `image_to_skeleton(bgr, detector) → (canvas, found)`

Converts a raw hand photo into a **224×224 white-on-black skeleton image**:

1. Resize image to 224×224 and convert BGR → RGB.
1. Run MediaPipe hand landmark detection (21 keypoints).
1. Compute a **tight bounding box** around the hand with 15% padding.
1. Draw the 21 **bone connections** (`_HAND_CONNECTIONS`) as white lines.
1. Draw each **joint** as a grey circle.
1. Returns the skeleton canvas and a `found` boolean.

> If no hand is detected, an all-black canvas is returned — these get cleaned up later by `remove_black_images()`.

______________________________________________________________________

### Dataset Pipeline

#### `download_dataset()`

Uses **`kagglehub`** to automatically download the [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) dataset. Requires `~/.kaggle/kaggle.json`.

#### `download_mediapipe_task(cfg)`

Downloads the `hand_landmarker.task` model file from Google's servers (skips if already present).

#### `build_skeleton_dataset(raw_dir, cfg)`

Iterates every class folder, calls `image_to_skeleton()` on each photo, and saves the result. Already-converted images are skipped (resumable). Progress is printed every 1,000 images.

#### `remove_black_images(skeleton_dir)`

Scans skeleton images using OpenCV and deletes any where the mean pixel value \< 2.0 — these are all-black frames where MediaPipe failed to detect a hand.

______________________________________________________________________

### Model Architecture — `build_model(num_classes)`

```
Input (224×224×3, values in [0,1])
  │
  ├─ Rescaling(255.0)          ← undo the /255 done by ImageDataGenerator
  ├─ Lambda(preprocess_input)  ← EfficientNetB3 expects [-1, 1]
  │
  └─ EfficientNetB3 backbone (frozen initially, ImageNet weights)
       │
       └─ GlobalAveragePooling2D
            └─ BatchNormalization
                 └─ Dense(512, relu)
                      └─ Dropout(0.4)
                           └─ Dense(256, relu)
                                └─ Dropout(0.3)
                                     └─ Dense(29, softmax, float32)
```

**Key design choices:**

- Preprocessing is **embedded inside the model** — inference just passes a `[0,1]` image and gets correct results automatically. No risk of preprocessing mismatch at deploy time.
- Output layer is forced to `float32` even in mixed-precision mode to avoid numerical instability in softmax.

______________________________________________________________________

### Data Generators — `make_generators(skeleton_dir, cfg)`

**Training augmentation** (tuned for white-on-black skeletons):

- Rotation ±20°
- Width/height shift ±12%
- Shear ±10%
- Zoom ±15%
- Horizontal flip (valid — ASL uses both hand orientations)
- Fill with black (matches skeleton background)

**Validation split:** 15% held out, no augmentation.

**Class weights:** `compute_class_weight("balanced")` ensures rare or hard sign classes (e.g., similar-looking letters) get equal attention during training.

______________________________________________________________________

### Learning Rate Schedule — `cosine_schedule(initial_lr, epochs, warmup)`

Uses **cosine decay with linear warmup**:

- First `warmup` epochs: LR ramps linearly from 0 → `initial_lr`
- After warmup: LR follows a cosine curve down to ~0

This avoids large gradient updates at the start and prevents the model getting stuck in local minima at the end.

______________________________________________________________________

### 3-Phase Training — `train(skeleton_dir, cfg)`

| Phase       | Layers trainable              | LR     | Epochs | Purpose                                                    |
| ----------- | ----------------------------- | ------ | ------ | ---------------------------------------------------------- |
| **Phase 1** | Head only                     | `1e-3` | 15     | Train the new classification head while backbone is frozen |
| **Phase 2** | Top 50 backbone layers + head | `5e-5` | 20     | Fine-tune upper backbone layers for ASL-specific features  |
| **Phase 3** | All layers                    | `1e-5` | 10     | Final polish at very low LR                                |

All phases use:

- `ModelCheckpoint` — saves the best model (by `val_accuracy`)
- `EarlyStopping` — halts if no improvement for `patience` epochs, restores best weights
- `cosine_schedule` — smooth LR decay within each phase

______________________________________________________________________

### Evaluation — `evaluate(cfg, class_names, val_gen)`

Loads the **best saved checkpoint** and runs:

1. **Classification report** (precision, recall, F1 per class)
1. **Confusion matrix** heatmap saved as `output_v2/confusion_matrix.png`

______________________________________________________________________

### Training Curves — `plot_curves(histories, cfg)`

Stitches together the three phase histories and plots:

- Train/validation accuracy over all epochs
- Train/validation loss over all epochs
- Vertical lines marking phase boundaries

Saved as `output_v2/training_curves.png`.

______________________________________________________________________

## Output Files

| File                                        | Description                            |
| ------------------------------------------- | -------------------------------------- |
| `output_v2/sign_language_model_best.keras`  | Best checkpoint (highest val accuracy) |
| `output_v2/sign_language_model_final.keras` | Model after all training phases        |
| `output_v2/class_names.json`                | List of 29 class labels in index order |
| `output_v2/confusion_matrix.png`            | Per-class accuracy heatmap             |
| `output_v2/training_curves.png`             | Accuracy & loss curves                 |
| `output_v2/skeleton_dataset/`               | Pre-processed skeleton images          |

## Deploying the Model

```bash
cp output_v2/sign_language_model_best.keras src/app/core/ml/trained_model/sign_language_mobilenet.keras
cp output_v2/class_names.json src/app/core/ml/trained_model/class_names.json
```
