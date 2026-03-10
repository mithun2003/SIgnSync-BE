"""PyTorch MLP model for ASL sign language classification (MediaPipe landmarks)."""

import torch
import torch.nn as nn

from .config import DROPOUT1, DROPOUT2, HIDDEN1, HIDDEN2, INPUT_DIM, NUM_CLASSES


class SignLanguageMLP(nn.Module):
    """63-input MLP for hand-landmark sign classification.

    Architecture:
        Input  (63)  → FC(256) → BN → ReLU → Dropout(0.3)
        Hidden (256) → FC(128) → BN → ReLU → Dropout(0.2)
        Output (128) → FC(33)  → LogSoftmax

    Parameters: ~85 K  |  Inference: <1 ms on CPU
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden1: int = HIDDEN1,
        hidden2: int = HIDDEN2,
        num_classes: int = NUM_CLASSES,
        dropout1: float = DROPOUT1,
        dropout2: float = DROPOUT2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout1),
            # Layer 2
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
            # Output
            nn.Linear(hidden2, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
