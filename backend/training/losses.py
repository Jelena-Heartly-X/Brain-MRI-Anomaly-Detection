"""
training/losses.py — Loss functions for Conv-MAE reconstruction training.

Implements:
  - MSE loss (pixel-wise)
  - SSIM loss (structural similarity)
  - L1 loss
  - Perceptual loss (VGG-based feature matching, optional)
  - Combined loss: 0.5 * MSE + 0.5 * (1 - SSIM) + optional extras
  - Multi-scale reconstruction loss with deep supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

try:
    from pytorch_msssim import ssim, ms_ssim, SSIM
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False
    print("[Loss] pytorch-msssim not found. Using manual SSIM fallback.")


# ─── SSIM (manual fallback) ───────────────────────────────────────────────────

def gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    x = torch.arange(size).float() - size // 2
    kernel = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def ssim_loss_manual(pred: torch.Tensor, target: torch.Tensor,
                     window_size: int = 11, sigma: float = 1.5,
                     C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    """Manual SSIM computation for single-channel images."""
    k1d = gaussian_kernel_1d(window_size, sigma).to(pred.device)
    k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)  # (W, W)
    k2d = k2d.expand(pred.shape[1], 1, window_size, window_size)

    pad = window_size // 2
    mu1 = F.conv2d(pred, k2d, padding=pad, groups=pred.shape[1])
    mu2 = F.conv2d(target, k2d, padding=pad, groups=pred.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, k2d, padding=pad, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target * target, k2d, padding=pad, groups=pred.shape[1]) - mu2_sq
    sigma12   = F.conv2d(pred * target, k2d, padding=pad, groups=pred.shape[1]) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-8)
    return 1.0 - ssim_map.mean()


# ─── Perceptual Loss (VGG) ────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    VGG16-based perceptual loss.
    Compares feature maps at multiple scales rather than pixel values.
    Only enabled when config.loss_perceptual_weight > 0.
    """

    def __init__(self, feature_layers: List[int] = [3, 8, 15, 22]):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(pretrained=True).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        self.slices = nn.ModuleList()
        prev = 0
        for layer in sorted(feature_layers):
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:layer + 1]))
            prev = layer + 1

        # Register buffer so it moves with model device
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert grayscale [0,1] to ImageNet-normalized RGB."""
        x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_vgg   = self._preprocess(pred)
        target_vgg = self._preprocess(target)
        loss = 0.0
        p, t = pred_vgg, target_vgg
        for sl in self.slices:
            p = sl(p)
            t = sl(t)
            loss = loss + F.l1_loss(p, t)
        return loss


# ─── Main Combined Loss ───────────────────────────────────────────────────────

class ConvMAELoss(nn.Module):
    """
    Combined reconstruction loss for Conv-MAE.

    Main loss:  alpha * MSE  +  beta * (1 - SSIM)
    Optional:   + gamma * L1  +  delta * Perceptual

    Also handles:
      - Masked-only loss (compute loss only on masked patches)
      - Multi-scale auxiliary loss with decay weights
    """

    def __init__(self, config, device: torch.device = None):
        super().__init__()
        self.config = config
        self.mse_w   = config.loss_mse_weight
        self.ssim_w  = config.loss_ssim_weight
        self.l1_w    = config.loss_l1_weight
        self.perc_w  = config.loss_perceptual_weight

        # Auxiliary scale weights (decay by 0.5 at each coarser scale)
        self.aux_weights = [0.5 ** (i + 1) for i in range(4)]

        if self.perc_w > 0:
            self.perceptual = PerceptualLoss()
        else:
            self.perceptual = None

        if HAS_MSSSIM:
            self.ssim_fn = SSIM(data_range=1.0, size_average=True, channel=1)

    def compute_ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if HAS_MSSSIM:
            return 1.0 - self.ssim_fn(pred, target)
        return ssim_loss_manual(pred, target)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        aux_preds: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred:      (B, 1, H, W) main reconstruction
            target:    (B, 1, H, W) original image
            mask:      (B, 1, H, W) binary mask (1 = was masked)
            aux_preds: list of (B, 1, H', W') auxiliary reconstructions

        Returns:
            total_loss, loss_dict
        """
        # ── Main reconstruction loss ──
        mse_loss  = F.mse_loss(pred, target)
        ssim_loss = self.compute_ssim_loss(pred, target)
        l1_loss   = F.l1_loss(pred, target)

        main_loss = self.mse_w * mse_loss + self.ssim_w * ssim_loss
        if self.l1_w > 0:
            main_loss = main_loss + self.l1_w * l1_loss

        # ── Masked-region emphasis ──
        if mask is not None:
            masked_mse  = (((pred - target) ** 2) * mask).sum() / (mask.sum() + 1e-6)
            main_loss = main_loss + 0.3 * masked_mse
        else:
            masked_mse = torch.tensor(0.0)

        # ── Perceptual loss ──
        perc_loss = torch.tensor(0.0, device=pred.device)
        if self.perc_w > 0 and self.perceptual is not None:
            perc_loss = self.perceptual(pred, target)
            main_loss = main_loss + self.perc_w * perc_loss

        # ── Auxiliary multi-scale losses ──
        aux_loss = torch.tensor(0.0, device=pred.device)
        if aux_preds:
            for aux_pred, w in zip(aux_preds, self.aux_weights):
                # Downsample target to match auxiliary output size
                t_down = F.interpolate(target, size=aux_pred.shape[-2:], mode="bilinear",
                                       align_corners=False)
                aux_loss = aux_loss + w * F.mse_loss(aux_pred, t_down)
            main_loss = main_loss + 0.2 * aux_loss

        loss_dict = {
            "total":    main_loss.item(),
            "mse":      mse_loss.item(),
            "ssim":     ssim_loss.item(),
            "l1":       l1_loss.item(),
            "perceptual": perc_loss.item() if torch.is_tensor(perc_loss) else perc_loss,
            "aux":      aux_loss.item(),
            "masked_mse": masked_mse.item() if torch.is_tensor(masked_mse) else 0.0,
        }

        return main_loss, loss_dict


# ─── Classifier Loss ──────────────────────────────────────────────────────────

class ClassifierLoss(nn.Module):
    """Binary cross-entropy with label smoothing for the anomaly classifier."""

    def __init__(self, label_smoothing: float = 0.05, pos_weight: float = 1.5):
        super().__init__()
        self.smoothing = label_smoothing
        self.pos_weight_val = pos_weight
        self.bce = nn.BCEWithLogitsLoss()

    def to(self, device):
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight_val], device=device)
        )
        return super().to(device)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = labels.float()
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)
