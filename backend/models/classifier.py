"""
models/classifier.py — Lightweight CNN classifier operating on reconstruction error maps.

Input:  concatenation of [original, reconstruction, error_map] → 3-channel input.
Output: probability of anomaly (binary classification).

Trained using pseudo-anomaly labels from PseudoAnomalyGenerator.
This provides an additional classification signal beyond thresholded error maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ErrorMapClassifier(nn.Module):
    """
    CNN classifier that takes [image, reconstruction, |image - recon|] as input
    and outputs probability of anomaly.

    Architecture: 4-stage progressively deepening CNN with global average pooling.
    """

    def __init__(self, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: (3, 224, 224) → (32, 112, 112)
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Stage 2: (32, 112, 112) → (64, 56, 56)
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Stage 3: (64, 56, 56) → (128, 28, 28)
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Stage 4: (128, 28, 28) → (256, 14, 14)
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Binary: normal vs anomaly
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image: torch.Tensor, recon: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: (B, 1, H, W) original MRI
            recon: (B, 1, H, W) Conv-MAE reconstruction

        Returns:
            dict with 'logit', 'prob', 'error_map'
        """
        error_map = (image - recon).abs()
        x = torch.cat([image, recon, error_map], dim=1)  # (B, 3, H, W)
        x = self.features(x)
        x = self.gap(x)
        logit = self.classifier(x).squeeze(1)  # (B,)
        prob = torch.sigmoid(logit)
        return {
            "logit": logit,
            "prob": prob,
            "error_map": error_map,
        }

    def predict_proba(self, image: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """Return anomaly probability without gradient computation."""
        with torch.no_grad():
            return self(image, recon)["prob"]
