# `train.ipynb` — Explained for the Team

> **Purpose:** The original notebook-based training script for the SignSync ASL recognition model. Uses **MobileNetV2** with a 2-phase training approach. This is what produced the currently deployed model before `train_v2.py` was written.

______________________________________________________________________

## What This Notebook Does

The notebook (`train.ipynb`) is a self-contained training pipeline for a 29-class ASL hand-sign classifier. It:

1. Downloads the ASL Alphabet dataset from Kaggle
1. Converts raw hand photos into MediaPipe skeleton images
1. Cleans up failed detections
1. Trains a MobileNetV2-based model in two phases
1. Evaluates the model and saves outputs

> **Note:** The notebook code mirrors `train_local.py` which is the extracted `.py` version of the same logic.

______________________________________________________________________

## How to Run

The notebook supports two modes via command-line arguments (or by editing the `Config` class directly):

```bash
# Quick test — 5 classes, 50 imgs/class
python train_local.py --test

# Full training — 29 classes, all images
python train_local.py --full
```

Or open `train.ipynb` in Jupyter and run all cells.

**Requirements:**

```
pip install tensorflow kagglehub opencv-python mediapipe scikit-learn matplotlib seaborn
```

Kaggle API token needed at `~/.kaggle/kaggle.json`.

______________________________________________________________________

## Overall Flow

```
main()
  │
  ├─ 1. setup_gpu()                      — GPU detection
  ├─ 2. download_dataset()               — Kaggle ASL Alphabet
  ├─ 3. download_mediapipe_model()       — hand_landmarker.task file
  ├─ 4. generate_skeleton_dataset()      — raw photos → skeleton images
  ├─ 5. cleanup_black_images()           — remove failed detections
  ├─ 6. check_class_distribution()       — print per-class counts & balance
  ├─ 7. train_model()                    — 2-phase MobileNetV2 training
  └─ 8. evaluate_model()                 — confusion matrix + report
```

______________________________________________________________________

## Cell-by-Cell / Function Breakdown

### Cell 0 — Docstring & Imports

Sets up the script description and imports all necessary libraries:

- `tensorflow` / `keras` — model building and training
- `cv2` (OpenCV) — image reading and skeleton drawing
- `mediapipe` — hand landmark detection
- `kagglehub` — dataset download
- `sklearn` — metrics (confusion matrix, classification report)
- `matplotlib` / `seaborn` — visualisation

______________________________________________________________________

### `setup_gpu()`

Checks for available GPUs and enables **memory growth** — prevents TensorFlow from allocating all available GPU memory at once, allowing other processes to share the GPU. Falls back gracefully to CPU.

______________________________________________________________________

### `Config` class

A simple configuration container. Two modes:

| Setting            | `--test` | `--full`       |
| ------------------ | -------- | -------------- |
| `batch_size`       | 8        | 32             |
| `epochs_phase1`    | 2        | 15             |
| `epochs_phase2`    | 2        | 20             |
| `max_classes`      | 5        | All 29         |
| `max_imgs_per_cls` | 50       | All (~3k each) |

Output paths are all rooted at `output/`.

______________________________________________________________________

### Skeleton Generation (MediaPipe)

#### `build_detector(task_path)`

Creates a **MediaPipe HandLandmarker** for single-image detection. Also populates the global `HAND_CONNECTIONS` list from the MediaPipe API (so it stays in sync with whatever MediaPipe version is installed — unlike `train_v2.py` which hard-codes the connections).

#### `image_to_skeleton(img_bgr, detector) → (canvas, found)`

Converts a raw hand photo into a **224×224 white-on-black skeleton image**:

1. Resize to 224×224, convert BGR → RGB.
1. Detect 21 hand landmarks with MediaPipe.
1. Build a tight bounding box around the hand with 15% padding.
1. Draw bone connections as white lines (thickness=2).
1. Draw joint circles in light grey (radius=4).
1. Returns `(canvas, True)` if a hand was found, or `(blank_canvas, False)` otherwise.

______________________________________________________________________

### Dataset Pipeline

#### `download_dataset()`

Uses **`kagglehub`** to download [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). Includes a friendly error message with step-by-step instructions if the Kaggle token is missing.

Returns the path to the `asl_alphabet_train` folder.

#### `download_mediapipe_model(task_path, task_url)`

Downloads `hand_landmarker.task` from Google's CDN. Skips download if file already exists.

#### `generate_skeleton_dataset(raw_dir, out_dir, config, task_path)`

Iterates every class folder and every image file:

- Skips images already converted (resumable pipeline).
- Calls `image_to_skeleton()` on each photo.
- Saves the skeleton PNG to the corresponding output class folder.
- Prints progress every 500 images.
- Reports `found_rate` (% of images where MediaPipe detected a hand) at the end.

#### `cleanup_black_images(skeleton_dir)`

Scans all `.jpg` files. Removes any image where:

- The file cannot be read (corrupted), OR
- The mean pixel value \< 5 (essentially all black — MediaPipe found no hand)

Reports how many images were kept vs. removed.

______________________________________________________________________

### Class Distribution Check — `check_class_distribution(skeleton_dir)`

Prints a per-class image count table. Flags classes with fewer than 500 images with a ⚠️ warning. Also reports the imbalance ratio (min/max count). If ratio \< 0.5, warns that the model may be biased toward over-represented classes.

______________________________________________________________________

### Model Architecture (inside `train_model`)

```
MobileNetV2 backbone (ImageNet weights, frozen initially)
  │
  └─ GlobalAveragePooling2D
       └─ Dense(256, relu)
            └─ Dropout(0.5)
                 └─ Dense(29, softmax)
```

**Key points:**

- MobileNetV2 is a lightweight backbone (3.4M params) — fast to train and deploy.
- The head is small and simple: one dense layer + dropout.
- Input images are scaled to `[0, 1]` by `ImageDataGenerator(rescale=1/255)`.

> ⚠️ **Known issue (fixed in `train_v2.py`):** MobileNetV2's `preprocess_input` expects `[-1, 1]` but this script feeds `[0, 1]`. This mismatch partially undermines the ImageNet transfer learning. `train_v2.py` embeds the correct preprocessing inside the model to fix this.

______________________________________________________________________

### Data Augmentation (inside `train_model`)

```python
ImageDataGenerator(
    rescale=1 / 255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)
```

20% of the data is held out for validation. No augmentation is applied at validation time.

______________________________________________________________________

### 2-Phase Training — `train_model(skeleton_dir, config)`

| Phase       | Layers trainable              | LR     | Epochs | Purpose                                            |
| ----------- | ----------------------------- | ------ | ------ | -------------------------------------------------- |
| **Phase 1** | Head only                     | `1e-3` | 15     | Train the new classification head, backbone frozen |
| **Phase 2** | Top 30 backbone layers + head | `1e-5` | 20     | Fine-tune upper MobileNetV2 layers                 |

**Callbacks used:**

- `EarlyStopping(patience=5)` — stops training if val accuracy doesn't improve, restores best weights
- `ReduceLROnPlateau(factor=0.5, patience=3)` — halves LR if val loss plateaus
- `ModelCheckpoint` — saves the best model by val accuracy

> Compare to `train_v2.py` which uses 3 phases and cosine LR decay instead of ReduceLROnPlateau.

______________________________________________________________________

### Evaluation — `evaluate_model(skeleton_dir, config, class_names)`

Loads the best saved model checkpoint and runs inference on the validation set:

1. **Classification report** — precision, recall, and F1-score per class
1. **Confusion matrix** heatmap — saved as `output/confusion_matrix.png`

______________________________________________________________________

## Output Files

| File                                     | Description                            |
| ---------------------------------------- | -------------------------------------- |
| `output/sign_language_model_best.keras`  | Best checkpoint (highest val accuracy) |
| `output/sign_language_model_final.keras` | Model after all training               |
| `output/class_names.json`                | List of class labels in index order    |
| `output/confusion_matrix.png`            | Per-class accuracy heatmap             |
| `output/skeleton_dataset/`               | Pre-processed skeleton images          |

______________________________________________________________________

## Comparison: `train.ipynb` vs `train_v2.py`

| Aspect                | `train.ipynb`                      | `train_v2.py`                        |
| --------------------- | ---------------------------------- | ------------------------------------ |
| **Backbone**          | MobileNetV2 (~3.4M params)         | EfficientNetB3 (~12M params)         |
| **Training phases**   | 2 (frozen → fine-tune top 30)      | 3 (frozen → fine-tune top 50 → full) |
| **Preprocessing**     | External (`/255` only, mismatched) | Embedded inside model (correct)      |
| **LR schedule**       | ReduceLROnPlateau                  | Cosine decay with warmup             |
| **Label smoothing**   | No                                 | Yes (0.1)                            |
| **Class weighting**   | No                                 | Yes (`balanced`)                     |
| **Val split**         | 20%                                | 15%                                  |
| **Expected accuracy** | ~80–85%                            | ≥ 95%                                |

Use `train_v2.py` for new training runs. `train.ipynb` is kept for reference and historical context.
