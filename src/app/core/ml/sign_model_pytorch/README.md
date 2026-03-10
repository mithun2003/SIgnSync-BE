# sign_model_pytorch — PyTorch/SVM Sign Language Model

A Windows-GPU-compatible rewrite of the sign language recognition system.

**Primary model: SVM on MediaPipe landmarks** — more accurate than CNN for
hand-pose classification because it works on clean, normalised 3-D joint
coordinates instead of raw pixels (no background clutter, no lighting issues,
no skeleton quality problems).

The CNN (EfficientNet-B0) is included as an optional alternative.

______________________________________________________________________

## What this model recognises

| Group        | Signs                         | Data source                             |
| ------------ | ----------------------------- | --------------------------------------- |
| ASL alphabet | A – Z (26)                    | Kaggle `grassknoted/asl-alphabet`       |
| Utility      | `space`, `del`                | Kaggle dataset                          |
| Emergency    | `help`, `danger`, `emergency` | **Synthetic** (generated automatically) |
| **Total**    | **31**                        |                                         |

### Emergency sign gestures

| Sign        | Gesture                                               | Why distinctive                                 |
| ----------- | ----------------------------------------------------- | ----------------------------------------------- |
| `help`      | Open flat palm — all 5 fingers spread, facing camera  | Maximum finger extension; not in A–Z            |
| `danger`    | ILY — index + pinky + thumb out, middle + ring curled | Unique 3-finger combination; not in A–Z         |
| `emergency` | Thumbs-up fist — thumb pointing straight UP           | Thumb direction differs from ASL 'A' (sideways) |

**No webcam data collection required** — synthetic landmark data is generated
automatically by `train_svm.py`.

______________________________________________________________________

## Quick start

### 1  Install dependencies

```bash
pip install -r requirements.txt
```

For Windows with CUDA 11.8 (GPU acceleration for CNN — optional):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

______________________________________________________________________

### 2  Train the SVM  *(recommended, no GPU needed)*

```bash
# Downloads ASL dataset from Kaggle, generates synthetic emergency landmarks, trains SVM
python train_svm.py

# Quick smoke-test (3 ASL classes only, ~2 min)
python train_svm.py --test

# Skip re-download if dataset already present
python train_svm.py --skip-download

# Use a local copy of the ASL dataset
python train_svm.py --data-dir /path/to/asl_alphabet_train
```

Output saved to `trained_model/`:

```
trained_model/
  sign_language_svm.joblib    ← trained SVM pipeline
  class_names_svm.json        ← ordered class list
```

Expected accuracy: **≥ 95 %** on the held-out test set.

**Why SVM beats CNN here:**

- Landmarks are pose-only features — zero background/lighting noise
- RBF SVM excels at high-dimensional normalised feature vectors
- Training takes **seconds** vs hours for CNN
- No GPU needed

______________________________________________________________________

### 2b  Train CNN  *(optional, GPU recommended)*

```bash
python train.py               # full training
python train.py --test        # quick smoke-test
python train.py --skip-prep   # data already prepared
```

______________________________________________________________________

### 3  Run inference

**Live webcam:**

```bash
python predict.py --webcam
python predict.py --webcam --model cnn
```

**Single image:**

```bash
python predict.py --image path/to/hand.jpg
```

**In Python (API drop-in):**

```python
from predict import predict_sign, load_ml_model, get_health_status, get_public_info

load_ml_model()  # optional — lazy-loads on first predict_sign call

result = predict_sign(raw_image_bytes)
# → {"label": "A", "confidence": 98.45}
```

### ⚠ API integration note

The new `predict_sign()` accepts **raw camera frames** (unprocessed JPEG/PNG).
MediaPipe runs server-side — no skeleton pre-processing on the frontend needed.

If your API currently sends skeleton images, update the frontend to send raw
frames, or add an intermediate step in the route handler.

______________________________________________________________________

## File layout

```
sign_model_pytorch/
├── config.py               ← class lists, sign definitions, paths, hyperparameters
├── model.py                ← EfficientNet-B0 CNN (optional)
├── dataset.py              ← dataset helpers and transforms (for CNN)
├── utils.py                ← MediaPipe skeleton + landmark extraction
├── train_svm.py            ← PRIMARY: SVM training with synthetic emergency data
├── train.py                ← OPTIONAL: CNN training (3-phase EfficientNet-B0)
├── collect_custom_data.py  ← OPTIONAL: webcam capture to improve emergency signs
├── predict.py              ← inference (SVM default, CNN fallback)
├── requirements.txt
├── data/
│   ├── skeleton/           ← generated by train.py (CNN only)
│   └── custom/             ← optional captured frames (collect_custom_data.py)
└── trained_model/          ← saved model files (created after training)
```

______________________________________________________________________

## Adding more emergency signs

1. Add sign name to `EMERGENCY_CLASSES` in `config.py`
1. Add gesture description to `EMERGENCY_SIGN_INSTRUCTIONS`
1. Add the canonical 21×3 landmark array to `_SYNTHETIC_SIGNS` in `train_svm.py`
1. Retrain: `python train_svm.py --skip-download`
