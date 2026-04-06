"""
models/conv_mae.py — Convolutional Masked Autoencoder (Conv-MAE).

Implements:
  - Patch-based masking (75% mask ratio, configurable)
  - Block-wise vs random masking strategies
  - Full 2D Conv-MAE and 3D Conv-MAE3D variants
  - Forward pass returns: reconstruction, mask, multi-scale auxiliary reconstructions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

from models.encoder import ConvMAEEncoder, ConvMAEEncoder3D
from models.decoder import ConvMAEDecoder, ConvMAEDecoder3D


# ─── Masking Strategies ───────────────────────────────────────────────────────

def random_patch_mask(batch_size: int, img_size: int, patch_size: int,
                      mask_ratio: float, device: torch.device) -> torch.Tensor:
    """
    Generate random patch masks.
    Returns binary mask (B, 1, H, W) where 1 = masked (hidden).
    """
    n_patches_h = img_size // patch_size
    n_patches_w = img_size // patch_size
    n_patches = n_patches_h * n_patches_w
    n_masked = int(n_patches * mask_ratio)

    masks = torch.zeros(batch_size, n_patches, device=device)
    for b in range(batch_size):
        idx = torch.randperm(n_patches, device=device)[:n_masked]
        masks[b, idx] = 1.0

    # Reshape to spatial mask
    masks = masks.view(batch_size, 1, n_patches_h, n_patches_w)
    masks = F.interpolate(masks, size=(img_size, img_size), mode="nearest")
    return masks


def block_patch_mask(batch_size: int, img_size: int, patch_size: int,
                     mask_ratio: float, device: torch.device) -> torch.Tensor:
    """
    Block-wise masking: masks contiguous rectangular blocks of patches.
    More realistic: forces model to reconstruct coherent regions.
    """
    n_h = img_size // patch_size
    n_w = img_size // patch_size
    n_masked = int(n_h * n_w * mask_ratio)

    masks = []
    for _ in range(batch_size):
        patch_mask = torch.zeros(n_h, n_w, device=device)
        remaining = n_masked
        while remaining > 0:
            # Random block size
            bh = np.random.randint(1, max(2, n_h // 2))
            bw = np.random.randint(1, max(2, n_w // 2))
            y = np.random.randint(0, n_h - bh + 1)
            x = np.random.randint(0, n_w - bw + 1)
            patch_mask[y:y+bh, x:x+bw] = 1.0
            remaining = max(0, n_masked - int(patch_mask.sum()))
        masks.append(patch_mask)

    masks = torch.stack(masks).unsqueeze(1)  # (B, 1, n_h, n_w)
    masks = F.interpolate(masks, size=(img_size, img_size), mode="nearest")
    return masks


def apply_mask(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out masked regions. mask=1 → hidden, mask=0 → visible."""
    return image * (1.0 - mask)


# ─── Conv-MAE (2D) ────────────────────────────────────────────────────────────

class ConvMAE(nn.Module):
    """
    2D Convolutional Masked Autoencoder for brain MRI self-supervised learning.

    Training: input is masked → encoder + decoder reconstruct full image.
    Inference: compute pixel-wise reconstruction error as anomaly score.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ConvMAEEncoder(config)
        self.decoder = ConvMAEDecoder(config)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def generate_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Generate masking tensor based on config."""
        B, C, H, W = x.shape
        if self.config.block_masking:
            return block_patch_mask(B, H, self.config.patch_size,
                                    self.config.mask_ratio, x.device)
        return random_patch_mask(B, H, self.config.patch_size,
                                 self.config.mask_ratio, x.device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x:    (B, 1, H, W) input image
            mask: optional pre-computed mask. If None, generate one.
        Returns:
            recon:     (B, 1, H, W) full reconstruction
            mask:      (B, 1, H, W) applied mask
            aux_recons: list of auxiliary multi-scale reconstructions
        """
        if mask is None:
            mask = self.generate_mask(x)

        masked_x = apply_mask(x, mask)
        latent, skips = self.encoder(masked_x)
        recon, aux_recons = self.decoder(latent, skips)

        return recon, mask, aux_recons

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass without masking — for anomaly inference.
        We pass the original image through without any masking, then
        compute reconstruction error.
        """
        latent, skips = self.encoder(x)
        recon, _ = self.decoder(latent, skips)
        return recon

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns pixel-wise MSE error map for anomaly detection.
        No masking applied during inference.
        """
        with torch.no_grad():
            recon = self.reconstruct(x)
            error = (x - recon).pow(2)
        return error

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Conv-MAE 3D ──────────────────────────────────────────────────────────────

class ConvMAE3D(nn.Module):
    """3D variant of Conv-MAE operating on full MRI volumes."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ConvMAEEncoder3D(config)
        self.decoder = ConvMAEDecoder3D(config)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _generate_volume_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Simple random 3D patch masking."""
        B, C, D, H, W = x.shape
        p = self.config.patch_size
        mask = torch.zeros(B, 1, D, H, W, device=x.device)
        n_patches = (D // p) * (H // p) * (W // p)
        n_masked = int(n_patches * self.config.mask_ratio)
        for b in range(B):
            for _ in range(n_masked):
                dz = np.random.randint(0, D - p + 1)
                dy = np.random.randint(0, H - p + 1)
                dx = np.random.randint(0, W - p + 1)
                mask[b, 0, dz:dz+p, dy:dy+p, dx:dx+p] = 1.0
        return mask

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is None:
            mask = self._generate_volume_mask(x)
        masked_x = apply_mask(x, mask)
        latent, skips = self.encoder(masked_x)
        recon, aux = self.decoder(latent, skips)
        return recon, mask, aux

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        latent, skips = self.encoder(x)
        recon, _ = self.decoder(latent, skips)
        return recon


# ─── Model Factory ────────────────────────────────────────────────────────────

def build_model(config) -> nn.Module:
    """Build the appropriate model based on config."""
    if config.use_3d:
        model = ConvMAE3D(config)
    else:
        model = ConvMAE(config)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {'ConvMAE3D' if config.use_3d else 'ConvMAE'} | "
          f"Parameters: {n_params/1e6:.2f}M")
    return model
