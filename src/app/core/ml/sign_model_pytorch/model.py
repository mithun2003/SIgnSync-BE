"""EfficientNet-B0 based CNN for sign language classification (PyTorch)."""

import torch
import torch.nn as nn
from torchvision import models


class SignCNN(nn.Module):
    """Sign language classifier built on a pre-trained EfficientNet-B0 backbone.

    The classifier head replaces the original ImageNet head with:
        Dropout(0.4) → Linear(1280→512) → BN → ReLU → Dropout(0.3) → Linear(512→num_classes)

    Three freeze helpers support the recommended 3-phase training strategy:
        1. ``freeze_backbone()``        — only train the head
        2. ``unfreeze_last_n_blocks()`` — gradually unfreeze the backbone
        3. ``unfreeze_all()``           — full fine-tuning
    """

    def __init__(self, num_classes: int, dropout: float = 0.4) -> None:
        super().__init__()

        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features: int = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    # ── Freeze / unfreeze helpers ─────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Phase 1: freeze entire backbone, keep classifier trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int = 2) -> None:
        """Phase 2: unfreeze the last *n* feature blocks of the backbone."""
        # Keep classifier trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
        # Unfreeze last n blocks from backbone.features
        blocks = list(self.backbone.features.children())
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Phase 3: make every parameter trainable."""
        for param in self.parameters():
            param.requires_grad = True

    # ── Utility ───────────────────────────────────────────────────────────────

    def trainable_params(self) -> int:
        """Return the count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
