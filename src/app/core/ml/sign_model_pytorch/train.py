#!/usr/bin/env python3
"""
Sign Language CNN Trainer  (PyTorch / EfficientNet-B0)
======================================================

Trains a CNN on MediaPipe hand-skeleton images for:
  • 26 ASL alphabets  (A–Z)
  • space + del        (2 utility signs)
  • help / danger / emergency  (custom emergency signs)

Usage
-----
  # Download ASL dataset, generate skeletons, then train  (default)
  python train.py

  # Quick smoke-test: 3 ASL classes, 40 images each, ~5 min
  python train.py --test

  # Skip downloading / skeleton generation (data already prepared)
  python train.py --skip-prep

  # Only prepare data, do not train
  python train.py --prepare-only

  # Use a locally downloaded ASL dataset folder
  python train.py --data-dir /path/to/asl_alphabet_train

Output
------
  trained_model/
    sign_language_cnn.pth   – final model checkpoint
    best_model.pth          – best checkpoint (by val accuracy)
    class_names.json        – ordered list of class labels
    training_curves.png     – loss + accuracy plot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Local imports ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from config import (
    ASL_CLASSES,
    BATCH_SIZE,
    CUSTOM_DATA_DIR,
    DATA_DIR,
    EMERGENCY_CLASSES,
    IMAGE_SIZE,
    KAGGLE_DATASET,
    MODEL_DIR,
    NUM_WORKERS,
    PHASE1_EPOCHS,
    PHASE1_LR,
    PHASE2_EPOCHS,
    PHASE2_LR,
    PHASE3_EPOCHS,
    PHASE3_LR,
    RANDOM_SEED,
    SKELETON_DIR,
    VAL_SPLIT,
)
from dataset import SampleDataset, get_all_samples, get_transforms, train_val_split
from model import SignCNN
from utils import extract_skeleton

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────


def download_asl_dataset() -> Path:
    """Download the ASL alphabet dataset from Kaggle."""
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed.  Run:  pip install kagglehub")
        sys.exit(1)

    print(f"Downloading '{KAGGLE_DATASET}' from Kaggle …")
    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        return Path(path)
    except Exception as exc:
        print(f"Kaggle download failed: {exc}")
        print("Make sure ~/.kaggle/kaggle.json contains your API credentials.")
        sys.exit(1)


def find_asl_source(dataset_root: Path) -> Path:
    """Return the directory that directly contains per-class sub-folders (A/, B/, …)."""
    candidates = [
        dataset_root / "asl_alphabet_train" / "asl_alphabet_train",
        dataset_root / "asl_alphabet_train",
        dataset_root,
    ]
    for c in candidates:
        if c.is_dir() and ((c / "A").exists() or (c / "a").exists()):
            return c

    # Fallback: recursive search
    for d in dataset_root.rglob("A"):
        if d.is_dir():
            return d.parent

    raise FileNotFoundError(f"Could not find ASL class folders inside {dataset_root}")


def generate_skeletons(
    src_dir: Path,
    dst_dir: Path,
    classes: list[str],
    max_per_class: int | None = None,
    skip_existing: bool = True,
) -> None:
    """Convert raw images → skeleton images and save to *dst_dir/<class>/*.

    Args:
        src_dir:       Directory containing one sub-folder per class.
        dst_dir:       Where to write skeleton images.
        classes:       Which classes to process.
        max_per_class: Cap per class (for quick tests).
        skip_existing: Skip classes where ≥90 % of expected images exist.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    for cls in classes:
        # Try exact name, then UPPER, then lower
        src_cls_dir: Path | None = None
        for variant in [cls, cls.upper(), cls.lower()]:
            if (src_dir / variant).is_dir():
                src_cls_dir = src_dir / variant
                break

        if src_cls_dir is None:
            print(f"  [{cls}] Not found in {src_dir} — skipping.")
            continue

        out_dir = dst_dir / cls
        out_dir.mkdir(exist_ok=True)

        images = [p for p in src_cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if max_per_class:
            images = images[:max_per_class]

        existing = len(list(out_dir.glob("*.jpg")))
        if skip_existing and existing >= int(len(images) * 0.9):
            print(f"  [{cls}] Already processed ({existing} skeletons) — skipping.")
            continue

        ok = fail = 0
        for img_path in tqdm(images, desc=f"  {cls}", leave=False):
            out_path = out_dir / (img_path.stem + ".jpg")
            if skip_existing and out_path.exists():
                ok += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                fail += 1
                continue

            skeleton = extract_skeleton(img, size=IMAGE_SIZE)
            if skeleton is None:
                fail += 1
                continue

            cv2.imwrite(str(out_path), skeleton)
            ok += 1

        print(f"  [{cls}] {ok} ok  |  {fail} failed")


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────


def _train_one_epoch(
    model: SignCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
) -> tuple[float, float]:
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def _evaluate(
    model: SignCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def _run_phase(
    name: str,
    model: SignCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    class_weights: torch.Tensor | None,
    use_amp: bool,
) -> list[dict]:
    """Run one training phase and save the best checkpoint."""
    print(f"\n{'=' * 62}")
    print(f"  {name}")
    print(f"  Epochs: {epochs}  |  LR: {lr:.1e}  |  Trainable: {model.trainable_params():,}")
    print(f"{'=' * 62}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=0.1,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)
    scaler = torch.amp.GradScaler() if use_amp else None

    best_val_acc = 0.0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc = _evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        improved = va_acc > best_val_acc
        if improved:
            best_val_acc = va_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")

        marker = "  ✓ best" if improved else ""
        print(
            f"  Ep {epoch:3d}/{epochs}  "
            f"train {tr_loss:.4f}/{tr_acc:.1f}%  "
            f"val {va_loss:.4f}/{va_acc:.1f}%  "
            f"({elapsed:.0f}s){marker}"
        )
        history.append({"train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})

    print(f"\n  Phase best val accuracy: {best_val_acc:.2f}%")
    return history


def _plot_history(history: list[dict], out_dir: Path) -> None:
    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, [h["train_acc"] for h in history], label="Train")
    axes[0].plot(epochs, [h["val_acc"] for h in history], label="Validation")
    axes[0].set_title("Accuracy (%)"), axes[0].set_xlabel("Epoch")
    axes[0].legend(), axes[0].grid(True)

    axes[1].plot(epochs, [h["train_loss"] for h in history], label="Train")
    axes[1].plot(epochs, [h["val_loss"] for h in history], label="Validation")
    axes[1].set_title("Loss"), axes[1].set_xlabel("Epoch")
    axes[1].legend(), axes[1].grid(True)

    # Mark phase boundaries
    for ep, color in [
        (PHASE1_EPOCHS, "gray"),
        (PHASE1_EPOCHS + PHASE2_EPOCHS, "gray"),
    ]:
        for ax in axes:
            ax.axvline(x=ep, color=color, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = out_dir / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved training curves → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sign language CNN (PyTorch)")
    parser.add_argument("--test", action="store_true", help="Quick test: 3 ASL classes, 40 images each")
    parser.add_argument("--skip-prep", action="store_true", help="Skip data preparation (skeletons already generated)")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare data; do not train")
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Path to the raw ASL dataset (skips Kaggle download)"
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision (AMP)")
    args = parser.parse_args()

    # ── Directories ───────────────────────────────────────────────────────────
    for d in (DATA_DIR, MODEL_DIR, SKELETON_DIR, CUSTOM_DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}  |  AMP: {use_amp}")

    # ── Determine working class list ──────────────────────────────────────────
    if args.test:
        asl_subset = ["A", "B", "C"]
        max_per_class: int | None = 40
        print("QUICK TEST MODE — 3 ASL classes, 40 images each")
    else:
        asl_subset = ASL_CLASSES
        max_per_class = None

    # ── Data preparation ──────────────────────────────────────────────────────
    if not args.skip_prep:
        raw_dir = args.data_dir or download_asl_dataset()
        asl_src = find_asl_source(Path(raw_dir))
        print(f"\nASL source directory: {asl_src}")

        print(f"\nGenerating skeletons for {len(asl_subset)} ASL classes …")
        generate_skeletons(asl_src, SKELETON_DIR, asl_subset, max_per_class=max_per_class)

        print("\nLooking for custom emergency sign data …")
        for ec in EMERGENCY_CLASSES:
            ec_raw_dir = CUSTOM_DATA_DIR / ec
            if ec_raw_dir.is_dir() and any(ec_raw_dir.iterdir()):
                count = len(list(ec_raw_dir.glob("*.jpg")))
                print(f"  [{ec}] Found {count} raw frames — generating skeletons …")
                generate_skeletons(CUSTOM_DATA_DIR, SKELETON_DIR, [ec])
            else:
                print(f"  [{ec}] No data found.  Run collect_custom_data.py first, then re-run with --skip-prep.")

    if args.prepare_only:
        print("\nData preparation done (--prepare-only).  Exiting.")
        return

    # ── Determine final class list (only classes with skeleton data) ──────────
    classes = [
        c for c in (asl_subset + EMERGENCY_CLASSES) if (SKELETON_DIR / c).is_dir() and any((SKELETON_DIR / c).iterdir())
    ]

    if len(classes) < 2:
        print("ERROR: Need at least 2 classes with skeleton data.  Aborting.")
        sys.exit(1)

    print(f"\nTraining with {len(classes)} classes: {classes}\n")

    # ── Build datasets ────────────────────────────────────────────────────────
    all_samples = get_all_samples(SKELETON_DIR, classes, seed=RANDOM_SEED)
    train_samples, val_samples = train_val_split(all_samples, VAL_SPLIT)

    train_ds = SampleDataset(train_samples, transform=get_transforms("train", IMAGE_SIZE))
    val_ds = SampleDataset(val_samples, transform=get_transforms("val", IMAGE_SIZE))

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": (device.type == "cuda"),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")
    print(f"Batches/epoch: {len(train_loader)}")

    # ── Class weights (handles imbalanced data) ───────────────────────────────
    labels_arr = np.array([s[1] for s in train_samples])
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(classes)),
        y=labels_arr,
    )
    class_weights = torch.tensor(cw, dtype=torch.float32)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SignCNN(num_classes=len(classes)).to(device)
    all_history: list[dict] = []

    # ── Phase 1: Frozen backbone ──────────────────────────────────────────────
    model.freeze_backbone()
    history = _run_phase(
        "PHASE 1 — Classifier head only (backbone frozen)",
        model,
        train_loader,
        val_loader,
        PHASE1_EPOCHS,
        PHASE1_LR,
        device,
        class_weights,
        use_amp,
    )
    all_history.extend(history)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=device))

    # ── Phase 2: Unfreeze last 2 backbone blocks ──────────────────────────────
    model.unfreeze_last_n_blocks(2)
    history = _run_phase(
        "PHASE 2 — Fine-tune last 2 EfficientNet blocks",
        model,
        train_loader,
        val_loader,
        PHASE2_EPOCHS,
        PHASE2_LR,
        device,
        class_weights,
        use_amp,
    )
    all_history.extend(history)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=device))

    # ── Phase 3: Full fine-tuning ─────────────────────────────────────────────
    model.unfreeze_all()
    history = _run_phase(
        "PHASE 3 — Full fine-tuning",
        model,
        train_loader,
        val_loader,
        PHASE3_EPOCHS,
        PHASE3_LR,
        device,
        class_weights,
        use_amp,
    )
    all_history.extend(history)
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=device))

    # ── Save artifacts ────────────────────────────────────────────────────────
    final_path = MODEL_DIR / "sign_language_cnn.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": classes,
            "num_classes": len(classes),
            "image_size": IMAGE_SIZE,
        },
        final_path,
    )

    cn_path = MODEL_DIR / "class_names.json"
    with open(cn_path, "w") as f:
        json.dump(classes, f, indent=2)

    _plot_history(all_history, MODEL_DIR)

    print(f"\n{'=' * 62}")
    print("  Training complete!")
    print(f"  Model  → {final_path}")
    print(f"  Labels → {cn_path}")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
