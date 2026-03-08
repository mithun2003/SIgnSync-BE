# `train.ipynb` vs `train_v2.py` — Side-by-Side Comparison

> A complete technical comparison of the two training scripts for the SignSync ASL hand-sign recognition model.
> Use this document to understand **what changed, why it changed, and which script to use**.

______________________________________________________________________

## TL;DR

|                           | `train.ipynb` / `train_local.py` | `train_v2.py`                         |
| ------------------------- | -------------------------------- | ------------------------------------- |
| **Status**                | Legacy — kept for reference      | ✅ Current — use for all new training |
| **Backbone**              | MobileNetV2                      | EfficientNetB3                        |
| **Expected val accuracy** | ~80–85%                          | ≥ 95%                                 |
| **Training phases**       | 2                                | 3                                     |
| **Preprocessing**         | ❌ Mismatched                    | ✅ Embedded & correct                 |
| **LR schedule**           | ReduceLROnPlateau                | Cosine decay + warmup                 |
| **Label smoothing**       | ❌ No                            | ✅ Yes (0.1)                          |
| **Class weighting**       | ❌ No                            | ✅ Yes                                |

______________________________________________________________________

## 1. Model Backbone

### `train.ipynb` — MobileNetV2

```python
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
```

- ~3.4M parameters total
- Designed for mobile/edge devices — fast but less expressive
- Top-1 ImageNet accuracy: ~71%

### `train_v2.py` — EfficientNetB3

```python
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=x)
```

- ~12M parameters total
- Compound scaling (depth + width + resolution) — significantly more accurate
- Top-1 ImageNet accuracy: ~81.6%
- 20–30% more accurate than MobileNetV2 on fine-grained classification tasks like ASL

**Why it matters:** ASL has 29 classes with many visually similar signs (e.g., A/E/S, R/U/V). A stronger backbone extracts richer spatial features that help distinguish these confusable pairs.

______________________________________________________________________

## 2. Input Preprocessing — The Critical Bug Fix

This is the **single biggest reason for the accuracy gap** between the two scripts.

### `train.ipynb` — Mismatched preprocessing ❌

```python
# ImageDataGenerator scales images to [0, 1]
datagen = ImageDataGenerator(rescale=1.0 / 255.0, ...)

# But MobileNetV2 was pretrained expecting [-1, 1]
# preprocess_input() is never called — transfer learning is broken
base = MobileNetV2(weights="imagenet", ...)
```

The network receives values in `[0, 1]` but its learned ImageNet features were computed on `[-1, 1]` inputs. The pretrained weights are essentially useless — the model trains from near-scratch.

### `train_v2.py` — Preprocessing embedded inside the model ✅

```python
# Model internally undoes /255, then applies EfficientNet's preprocess_input
x = layers.Rescaling(255.0)(inputs)          # [0,1] → [0,255]
x = layers.Lambda(preprocess_input)(x)       # [0,255] → [-1,1]
base = EfficientNetB3(..., input_tensor=x)
```

- The model **always receives the right input scale**, regardless of how the caller prepares images.
- No risk of preprocessing mismatch at inference/deploy time.
- Fully leverages the ImageNet pretrained weights.

______________________________________________________________________

## 3. Training Phases

### `train.ipynb` — 2-Phase Training

```
Phase 1 — Frozen backbone (15 epochs, LR=1e-3)
  └─ Only the Dense head is trained

Phase 2 — Fine-tune top 30 layers (20 epochs, LR=1e-5)
  └─ Top 30 MobileNetV2 layers + head unfrozen
```

### `train_v2.py` — 3-Phase Training

```
Phase 1 — Frozen backbone (15 epochs, LR=1e-3)
  └─ Only the Dense head is trained

Phase 2 — Fine-tune top 50 layers (20 epochs, LR=5e-5)
  └─ Top 50 EfficientNetB3 layers + head unfrozen

Phase 3 — Full fine-tune (10 epochs, LR=1e-5)
  └─ All layers unfrozen at very low LR for final polish
```

**Why 3 phases?** EfficientNetB3 is deeper and has more specialised early features worth preserving. The extra phase 3 gives the full network a chance to settle together without catastrophic forgetting.

______________________________________________________________________

## 4. Classification Head

### `train.ipynb`

```python
x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(29, activation="softmax")(x)
```

Simple 1-layer head.

### `train_v2.py`

```python
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.BatchNormalization()(x)
x = layers.Dense(512)(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.4)(x)
x = layers.Dense(256)(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.3)(x)
out = layers.Dense(29, activation="softmax", dtype="float32")(x)
```

- Deeper 2-layer head with BatchNorm for stability.
- Lower dropout (0.4/0.3 vs 0.5) — EfficientNetB3 features are already more generalised.
- `dtype="float32"` on the output layer prevents numerical instability with mixed precision.

______________________________________________________________________

## 5. Data Augmentation

### `train.ipynb`

```python
ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2,       # 20% val
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)
```

### `train_v2.py`

```python
ImageDataGenerator(
    rescale=1/255,
    validation_split=0.15,      # 15% val — more data for training
    rotation_range=20,          # slightly stronger rotation
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.10,           # ← new: shear transformation
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="constant", cval=0,  # ← fills new pixels with black (matches skeleton bg)
)
```

Key differences:

- **`shear_range`** added — helps the model handle slightly angled hand poses.
- **`fill_mode="constant", cval=0`** — new pixels after rotation/shift are filled black, matching the skeleton background. The notebook would fill with the nearest edge pixel, which creates unrealistic white smears on the skeleton.
- **Brightness range removed** — brightness variation doesn't make sense for white-on-black skeleton images; it was designed for natural photos.

______________________________________________________________________

## 6. Loss Function

### `train.ipynb`

```python
model.compile(loss="categorical_crossentropy", ...)
```

### `train_v2.py`

```python
model.compile(loss=CategoricalCrossentropy(label_smoothing=0.1), ...)
```

**Label smoothing (0.1):** Prevents the model from becoming overconfident on visually similar classes. Instead of training towards a hard `[0,0,1,0,...]` target, it trains towards `[0.003, 0.003, 0.9, 0.003, ...]`. This improves generalisation on hard letter pairs like A/E/S.

______________________________________________________________________

## 7. Class Imbalance Handling

### `train.ipynb`

No class weighting — the model sees all classes equally regardless of how many samples exist.

### `train_v2.py`

```python
class_weights = compute_class_weight("balanced", classes=..., y=train_gen.classes)
model.fit(..., class_weight=cw_dict)
```

Rare or hard classes get a proportionally higher loss contribution, preventing the model from being biased toward majority classes.

______________________________________________________________________

## 8. Learning Rate Schedule

### `train.ipynb` — ReduceLROnPlateau

```python
ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
```

Reactive: cuts LR in half when val loss stops improving. Can get stuck at suboptimal plateaus.

### `train_v2.py` — Cosine Decay with Warmup

```python
def cosine_schedule(initial_lr, epochs, warmup=2):
    # Linear warmup for first `warmup` epochs
    # Then smooth cosine decay to near-0
```

Proactive: LR follows a smooth cosine curve, warming up gradually then annealing. Avoids large updates early on and ensures the model converges smoothly at the end of each phase.

______________________________________________________________________

## 9. MediaPipe Hand Connections

### `train.ipynb`

```python
# Loaded dynamically from the MediaPipe API
HAND_CONNECTIONS = [(c.start, c.end) for c in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS]
```

Stays in sync with any MediaPipe version automatically.

### `train_v2.py`

```python
# Hard-coded as a fixed list of 21 tuples
_HAND_CONNECTIONS = [(0,1),(1,2),...,(0,17)]
```

Version-independent — won't break if MediaPipe changes its API, but must be manually updated if the hand topology changes (which is rare).

______________________________________________________________________

## 10. Hardware Optimisation

### `train.ipynb`

```python
# Memory growth only
tf.config.experimental.set_memory_growth(gpu, True)
```

### `train_v2.py`

```python
# Memory growth + mixed precision (float16)
mixed_precision.set_global_policy("mixed_float16")
```

Mixed precision (`float16`) roughly **doubles training throughput** on NVIDIA RTX 20xx+ GPUs with Tensor Cores, with no meaningful accuracy loss.

______________________________________________________________________

## 11. Validation Split

|                    | `train.ipynb` | `train_v2.py` |
| ------------------ | ------------- | ------------- |
| Val split          | 20%           | 15%           |
| More training data | ❌            | ✅            |

A 15% split gives 5% more images to the training set (~4,350 extra images on the full dataset) while still having a statistically significant validation set.

______________________________________________________________________

## 12. Output Locations

| Output           | `train.ipynb`                            | `train_v2.py`                               |
| ---------------- | ---------------------------------------- | ------------------------------------------- |
| Best model       | `output/sign_language_model_best.keras`  | `output_v2/sign_language_model_best.keras`  |
| Final model      | `output/sign_language_model_final.keras` | `output_v2/sign_language_model_final.keras` |
| Class names      | `output/class_names.json`                | `output_v2/class_names.json`                |
| Confusion matrix | `output/confusion_matrix.png`            | `output_v2/confusion_matrix.png`            |
| Skeleton dataset | `output/skeleton_dataset/`               | `output_v2/skeleton_dataset/`               |

______________________________________________________________________

## Summary — What Each Fix Contributes to Accuracy

| Fix                                        | Estimated accuracy gain |
| ------------------------------------------ | ----------------------- |
| EfficientNetB3 over MobileNetV2            | +8–12%                  |
| Correct preprocessing (biggest single fix) | +5–10%                  |
| 3-phase training vs 2-phase                | +2–4%                   |
| Label smoothing                            | +1–2%                   |
| Class weighting                            | +1–2%                   |
| Better augmentation (fill_mode, shear)     | +0.5–1%                 |
| Cosine LR vs ReduceLROnPlateau             | +0.5–1%                 |
| **Combined (with overlap)**                | **~15–20% total**       |

This is consistent with the observed jump from ~80% → ≥ 95% validation accuracy.

______________________________________________________________________

## Which Script Should I Use?

| Situation                           | Use                                            |
| ----------------------------------- | ---------------------------------------------- |
| Training a new model for deployment | `train_v2.py --full`                           |
| Quick sanity check / CI smoke test  | `train_v2.py --test`                           |
| Understanding the original model    | `train.ipynb`                                  |
| Debugging inference mismatch issues | Read `train_v2.py` — preprocessing is embedded |
| Reproducing the old deployed model  | `train.ipynb` / `train_local.py`               |
