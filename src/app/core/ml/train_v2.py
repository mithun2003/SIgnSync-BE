"""SignSync ASL Recognition — Training Script v2 (PyTorch)
═══════════════════════════════════════════════════════════════════
Trains a 29-class ASL model (A-Z, space, del, nothing) from the
Kaggle "grassknoted/asl-alphabet" dataset.

Converted from TensorFlow/Keras → PyTorch for native Windows GPU support.

USAGE
──────
    python train_v2_pytorch.py --test      # smoke test (~5 min)
    python train_v2_pytorch.py --full      # full training (~45 min GPU)

REQUIREMENTS
─────────────
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install kagglehub opencv-python mediapipe scikit-learn matplotlib seaborn
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
import torch
import torch.nn as nn
import torch.optim as optim
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B3_Weights

# ═══════════════════════════════════════════════════════════════════════════════
#  GPU SETUP
# ═══════════════════════════════════════════════════════════════════════════════


def setup_hardware() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU found: {gpu_name}  |  mixed_float16 (AMP) ON")
    else:
        device = torch.device("cpu")
        print("ℹ️  No GPU — running on CPU (slower but works)")
    return device


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


class Config:
    IMG_SIZE = (224, 224)
    INPUT_SHAPE = (224, 224, 3)
    NUM_CLASSES = 29

    OUTPUT_DIR = Path("output_v2")
    SKELETON_DIR = OUTPUT_DIR / "skeleton_dataset"
    MODEL_BEST = OUTPUT_DIR / "sign_language_model_best.pth"
    MODEL_FINAL = OUTPUT_DIR / "sign_language_model_final.pth"
    CLASS_JSON = OUTPUT_DIR / "class_names.json"
    TASK_FILE = OUTPUT_DIR / "hand_landmarker.task"
    TASK_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )

    def __init__(self, test_mode: bool = False) -> None:
        self.test_mode = test_mode
        self.OUTPUT_DIR.mkdir(exist_ok=True)

        if test_mode:
            self.batch_size = 8
            self.max_classes = 5
            self.max_imgs_per_class = 10
            self.epochs = (3, 3, 2)
            print("🧪 TEST MODE — 5 classes, 10 imgs/class, 3+3+2 epochs")
        else:
            self.batch_size = 32
            self.max_classes = None
            self.max_imgs_per_class = None
            self.epochs = (15, 20, 10)
            print("🚀 FULL MODE — 29 classes, all images, 15+20+10 epochs")


# ═══════════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE: SKELETON GENERATION  (unchanged from TF version)
# ═══════════════════════════════════════════════════════════════════════════════

_HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
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
    px = (x1 - x0) * 0.15
    py = (y1 - y0) * 0.15
    x0 -= px; x1 += px
    y0 -= py; y1 += py
    w = x1 - x0
    h = y1 - y0

    for s, e in _HAND_CONNECTIONS:
        p1, p2 = lms[s], lms[e]
        cv2.line(
            canvas,
            (_to_px(p1.x, x0, w), _to_px(p1.y, y0, h)),
            (_to_px(p2.x, x0, w), _to_px(p2.y, y0, h)),
            (255, 255, 255), 2,
        )
    for lm in lms:
        cv2.circle(
            canvas,
            (_to_px(lm.x, x0, w), _to_px(lm.y, y0, h)),
            4, (200, 200, 200), -1,
        )
    return canvas, True


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET DOWNLOAD & SKELETON PRE-GENERATION  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════


def download_dataset() -> str:
    print("\n" + "═" * 60)
    print("  STEP 1 — Downloading ASL Alphabet Dataset from Kaggle")
    print("═" * 60)
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
#  MODEL ARCHITECTURE  (PyTorch EfficientNetB3)
# ═══════════════════════════════════════════════════════════════════════════════


def build_model(num_classes: int) -> nn.Module:
    """EfficientNetB3 with a custom classification head."""
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    # Freeze backbone initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model         : EfficientNetB3 + custom head")
    print(f"  Total params  : {total:,}")
    print(f"  Trainable     : {trainable:,}  (head only — backbone frozen)")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS  (replaces ImageDataGenerator)
# ═══════════════════════════════════════════════════════════════════════════════


def make_dataloaders(skeleton_dir: Path, cfg: Config):
    """Return (train_loader, val_loader, class_names, class_weights_tensor)."""

    # EfficientNet pretrained on ImageNet — use official mean/std
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), shear=10, scale=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Load full dataset then split 85/15
    full_dataset = datasets.ImageFolder(str(skeleton_dir))
    class_names  = full_dataset.classes

    total = len(full_dataset)
    val_size   = int(total * 0.15)
    train_size = total - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform   = val_transform

    # Class weights for balanced sampling
    train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=train_labels,
    )
    cw_tensor = torch.FloatTensor(class_weights)

    # Weighted sampler so rare classes are seen equally often
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"\n  Train batches : {len(train_loader):,}  ({train_size:,} images)")
    print(f"  Val   batches : {len(val_loader):,}  ({val_size:,} images)")
    print(f"  Classes       : {class_names}")

    return train_loader, val_loader, class_names, cw_tensor


# ═══════════════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ═══════════════════════════════════════════════════════════════════════════════


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_prob  = nn.functional.log_softmax(logits, dim=-1)
        # Smooth target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_prob, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_targets * log_prob).sum(dim=-1).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  ONE EPOCH HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train() if train else model.eval()
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING  (3-phase: frozen → top-50 unfrozen → full)
# ═══════════════════════════════════════════════════════════════════════════════


def train_model(skeleton_dir: Path, cfg: Config, device: torch.device):
    print("\n" + "═" * 60)
    print("  STEP 4 — Building Model & Training")
    print("═" * 60)

    train_loader, val_loader, class_names, cw_tensor = make_dataloaders(skeleton_dir, cfg)
    model     = build_model(len(class_names)).to(device)
    criterion = LabelSmoothingCrossEntropy(0.1)
    scaler    = GradScaler()

    histories: dict[str, list] = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    phase_boundaries: list[int] = []
    best_val_acc = 0.0
    ep1, ep2, ep3 = cfg.epochs

    def _run_phase(phase_name: str, epochs: int, lr: float, unfreeze_top: int | None = None):
        nonlocal best_val_acc

        if unfreeze_top == 0:
            # Phase 1 — only head is trainable (already set in build_model)
            pass
        elif unfreeze_top is not None:
            # Phase 2 — unfreeze last N backbone layers
            backbone_layers = list(model.features.children())
            for block in backbone_layers[:-unfreeze_top]:
                for p in block.parameters():
                    p.requires_grad = False
            for block in backbone_layers[-unfreeze_top:]:
                for p in block.parameters():
                    p.requires_grad = True
        else:
            # Phase 3 — unfreeze everything
            for p in model.parameters():
                p.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'─' * 60}")
        print(f"  {phase_name}  (up to {epochs} epochs, LR={lr})  Trainable: {trainable:,}")
        print("─" * 60)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        patience_counter = 0
        patience_limit   = 6 if unfreeze_top == 0 else 8

        for ep in range(1, epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
            vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, scaler, device, train=False)
            scheduler.step()

            histories["train_acc"].append(tr_acc)
            histories["val_acc"].append(vl_acc)
            histories["train_loss"].append(tr_loss)
            histories["val_loss"].append(vl_loss)

            flag = ""
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save(model.state_dict(), str(cfg.MODEL_BEST))
                flag = "  ← best"
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"  Epoch {ep:3d}/{epochs}  "
                  f"loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
                  f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{flag}")

            if patience_counter >= patience_limit:
                print(f"  Early stopping (patience={patience_limit})")
                break

        phase_boundaries.append(len(histories["train_acc"]))

    _run_phase("PHASE 1 — Frozen backbone", ep1, 1e-3, unfreeze_top=0)
    _run_phase("PHASE 2 — Fine-tune top-3 blocks", ep2, 5e-5, unfreeze_top=3)
    _run_phase("PHASE 3 — Full fine-tune", ep3, 1e-5, unfreeze_top=None)

    print(f"\n✅ Training complete — best val accuracy: {best_val_acc * 100:.2f} %")

    # Save final model + class names
    torch.save(model.state_dict(), str(cfg.MODEL_FINAL))
    with open(cfg.CLASS_JSON, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"   Best model  → {cfg.MODEL_BEST}")
    print(f"   Final model → {cfg.MODEL_FINAL}")
    print(f"   Classes     → {cfg.CLASS_JSON}")

    return histories, phase_boundaries, class_names, model, val_loader


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate(cfg: Config, class_names: list[str], model: nn.Module, val_loader, device: torch.device) -> None:
    print("\n" + "═" * 60)
    print("  STEP 5 — Evaluation")
    print("═" * 60)

    # Load best weights
    model.load_state_dict(torch.load(str(cfg.MODEL_BEST), map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n📊 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    cm = confusion_matrix(all_labels, all_preds)
    n  = len(class_names)
    plt.figure(figsize=(max(14, n), max(11, n)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
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


def plot_curves(histories: dict, phase_boundaries: list[int], cfg: Config) -> None:
    acc     = histories["train_acc"]
    val_acc = histories["val_acc"]
    loss    = histories["train_loss"]
    val_loss= histories["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    boundary_labels = ["Fine-tune top blocks", "Full fine-tune"]

    for ax, train_m, val_m, title in [
        (ax1, acc, val_acc, "Accuracy"),
        (ax2, loss, val_loss, "Loss"),
    ]:
        ax.plot(epochs_range, train_m, "b-o", label="Train",      markersize=3)
        ax.plot(epochs_range, val_m,   "r-o", label="Validation", markersize=3)
        for i, b in enumerate(phase_boundaries[:-1]):
            ax.axvline(b, color=["gray", "orange"][i], linestyle="--", alpha=0.6,
                       label=boundary_labels[i])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Curves — EfficientNetB3, 3-Phase (PyTorch)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = cfg.OUTPUT_DIR / "training_curves.png"
    plt.savefig(str(out), dpi=120, bbox_inches="tight")
    print(f"✅ Training curves saved → {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Train improved ASL model (PyTorch)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true", help="Quick smoke test")
    mode.add_argument("--full", action="store_true", help="Full production training")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  SignSync ASL Training v2 — EfficientNetB3 (PyTorch)")
    print("═" * 60)

    device = setup_hardware()
    cfg    = Config(test_mode=args.test)

    # ── 1. Data ──
    if _skeleton_dataset_exists(cfg.SKELETON_DIR):
        print(f"\n✅ Skeleton dataset already exists at {cfg.SKELETON_DIR} — skipping conversion")
    else:
        raw_dir = download_dataset()
        download_mediapipe_task(cfg)
        build_skeleton_dataset(raw_dir, cfg)

    remove_black_images(cfg.SKELETON_DIR)

    # ── 2. Train ──
    histories, phase_boundaries, class_names, model, val_loader = train_model(cfg.SKELETON_DIR, cfg, device)

    # ── 3. Evaluate & plot ──
    evaluate(cfg, class_names, model, val_loader, device)
    plot_curves(histories, phase_boundaries, cfg)

    print("\n" + "═" * 60)
    print("  ALL DONE 🎉")
    print("═" * 60)
    print(f"\n  Best model  → {cfg.MODEL_BEST}")
    print(f"  Class names → {cfg.CLASS_JSON}")
    print()
    print("  ► Deploy:")
    print(f"    copy {cfg.MODEL_BEST} src\\app\\core\\ml\\trained_model\\sign_language_model.pth")
    print(f"    copy {cfg.CLASS_JSON} src\\app\\core\\ml\\trained_model\\class_names.json")


if __name__ == "__main__":
    main()