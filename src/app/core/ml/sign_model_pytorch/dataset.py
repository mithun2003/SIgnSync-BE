"""Dataset utilities for loading skeleton images and computing transforms."""

import random
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet statistics used for normalisation
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Type alias for a labelled sample
Sample = tuple[Path, int]


# ── Dataset class ─────────────────────────────────────────────────────────────


class SampleDataset(Dataset):
    """Minimal dataset that loads pre-generated skeleton images from a list of ``(path, label_index)`` tuples."""

    def __init__(
        self,
        samples: list[Sample],
        transform: Callable | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Sample loading ────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


def get_all_samples(
    data_dir: Path,
    class_names: list[str],
    seed: int = 42,
) -> list[Sample]:
    """Collect and shuffle all ``(image_path, class_index)`` pairs.

    Scans *data_dir/<class_name>/* for each class in *class_names*. Samples are shuffled with the given seed for
    reproducibility.
    """
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    samples: list[Sample] = []

    for cls in class_names:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in _IMG_EXTS:
                samples.append((img_path, class_to_idx[cls]))

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples


def train_val_split(
    samples: list[Sample],
    val_fraction: float = 0.15,
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into train and validation sets."""
    n_val = int(len(samples) * val_fraction)
    return samples[n_val:], samples[:n_val]


# ── Transforms ────────────────────────────────────────────────────────────────


def get_transforms(phase: str, image_size: int = 224) -> transforms.Compose:
    """Return torchvision transforms for *train* or *val* phase."""
    if phase == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
                transforms.ToTensor(),
                transforms.Normalize(_MEAN, _STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )
