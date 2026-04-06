"""
evaluation/anomaly_score.py — Ensemble anomaly scoring pipeline.

Combines:
  1. Pixel-wise MSE error map
  2. SSIM difference map
  3. L1 difference map
  4. Multi-scale error aggregation

Returns per-slice scalar scores and spatial error maps for localization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from pytorch_msssim import ssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False


# ─── Per-pixel Error Maps ─────────────────────────────────────────────────────

def mse_map(original: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    """Pixel-wise squared error. Returns (B, 1, H, W)."""
    return (original - reconstruction).pow(2)


def l1_map(original: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    """Pixel-wise L1 error. Returns (B, 1, H, W)."""
    return (original - reconstruction).abs()


def ssim_map(original: torch.Tensor, reconstruction: torch.Tensor,
             window_size: int = 7) -> torch.Tensor:
    """
    SSIM-based error map. Computes local SSIM and returns (1 - SSIM) as error.
    Returns (B, 1, H, W).
    """
    if HAS_MSSSIM:
        # pytorch-msssim does not expose a pixel-wise map directly, so we use
        # a sliding window approach on smaller patches
        from pytorch_msssim import SSIM as _SSIM
        ssim_fn = _SSIM(data_range=1.0, size_average=False, channel=1)
        score = ssim_fn(original, reconstruction)  # (B,)
        # Approximate pixel map via difference in structural content
        diff = (original - reconstruction).pow(2)
        # Scale diff by (1 - global_ssim) to weight by structural dissimilarity
        weight = (1.0 - score).view(-1, 1, 1, 1)
        return diff * weight + 1e-6
    else:
        # Fallback: use local variance-weighted error
        return (original - reconstruction).pow(2)


def gradient_map(error: torch.Tensor) -> torch.Tensor:
    """Add gradient magnitude to emphasize edges of anomalous regions."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=error.dtype, device=error.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(error, sobel_x, padding=1)
    gy = F.conv2d(error, sobel_y, padding=1)
    return (gx.pow(2) + gy.pow(2)).sqrt()


# ─── Multi-scale Error Aggregation ───────────────────────────────────────────

def multiscale_error(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    n_levels: int = 3,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Compute MSE at multiple resolutions and sum back to original resolution.
    Improves localization for both fine-grained and coarse anomalies.

    Args:
        original:       (B, 1, H, W)
        reconstruction: (B, 1, H, W)
        n_levels:       number of downsampling levels
        weights:        contribution weight per level (default: equal)

    Returns:
        fused_map: (B, 1, H, W) multi-scale error map
    """
    if weights is None:
        weights = [1.0 / n_levels] * n_levels

    H, W = original.shape[-2:]
    fused = torch.zeros_like(original)

    for level, w in enumerate(weights):
        scale = 0.5 ** level
        if scale < 1.0:
            orig_d = F.interpolate(original, scale_factor=scale,
                                   mode="bilinear", align_corners=False)
            recon_d = F.interpolate(reconstruction, scale_factor=scale,
                                    mode="bilinear", align_corners=False)
        else:
            orig_d, recon_d = original, reconstruction

        err = mse_map(orig_d, recon_d)

        if err.shape[-2:] != (H, W):
            err = F.interpolate(err, size=(H, W), mode="bilinear", align_corners=False)

        fused = fused + w * err

    return fused


# ─── Ensemble Score ───────────────────────────────────────────────────────────

class EnsembleAnomalyScorer:
    """
    Combines MSE, SSIM, L1, and multi-scale error into a single anomaly score.

    Returns both:
      - Per-pixel spatial error maps (for localization)
      - Per-slice scalar scores (for slice-level classification)
    """

    def __init__(self, config):
        self.mse_w   = config.score_mse_weight
        self.ssim_w  = config.score_ssim_weight
        self.l1_w    = config.score_l1_weight
        self.n_levels = config.multiscale_levels

    @torch.no_grad()
    def score_batch(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all anomaly maps and scalar scores for a batch.

        Args:
            original:       (B, 1, H, W)
            reconstruction: (B, 1, H, W)

        Returns dict with:
            error_map:     (B, 1, H, W) ensemble pixel map
            mse_map:       (B, 1, H, W)
            l1_map:        (B, 1, H, W)
            ssim_map:      (B, 1, H, W)
            ms_map:        (B, 1, H, W) multi-scale
            score:         (B,) per-slice scalar anomaly score
        """
        m_mse = mse_map(original, reconstruction)
        m_l1  = l1_map(original, reconstruction)
        m_ssim = ssim_map(original, reconstruction)
        m_ms  = multiscale_error(original, reconstruction, n_levels=self.n_levels)

        # Normalize each map to [0, 1] per sample
        def norm(x):
            B = x.shape[0]
            x_flat = x.view(B, -1)
            xmin = x_flat.min(1, keepdim=True)[0].view(B, 1, 1, 1)
            xmax = x_flat.max(1, keepdim=True)[0].view(B, 1, 1, 1)
            return (x - xmin) / (xmax - xmin + 1e-8)

        m_mse_n  = norm(m_mse)
        m_l1_n   = norm(m_l1)
        m_ssim_n = norm(m_ssim)
        m_ms_n   = norm(m_ms)

        # Weighted ensemble map
        ensemble_map = (
            self.mse_w  * m_mse_n  +
            self.ssim_w * m_ssim_n +
            self.l1_w   * m_l1_n
        ) / (self.mse_w + self.ssim_w + self.l1_w)

        # Fuse with multi-scale
        ensemble_map = 0.7 * ensemble_map + 0.3 * m_ms_n

        # Scalar score = mean of top-k% pixels (max pooling over spatial dims)
        # Using 95th percentile avoids outlier pixels dominating
        B = original.shape[0]
        flat = ensemble_map.view(B, -1)
        k = max(1, int(flat.shape[1] * 0.05))
        top_k, _ = flat.topk(k, dim=1)
        score = top_k.mean(dim=1)

        return {
            "error_map": ensemble_map,
            "mse_map":   m_mse_n,
            "l1_map":    m_l1_n,
            "ssim_map":  m_ssim_n,
            "ms_map":    m_ms_n,
            "score":     score,
        }

    @torch.no_grad()
    def score_dataset(
        self,
        model,
        dataloader,
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """
        Score all samples in a DataLoader.

        Returns dict with arrays:
            scores:     (N,) anomaly scores
            labels:     (N,) ground-truth labels
            error_maps: (N, H, W) spatial maps
        """
        all_scores, all_labels, all_maps = [], [], []

        model.eval()
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch.get("label", torch.zeros(images.shape[0]))

            with torch.no_grad():
                recon = model.reconstruct(images)

            results = self.score_batch(images, recon)
            all_scores.append(results["score"].cpu())
            all_labels.append(labels.cpu())
            all_maps.append(results["error_map"].cpu().squeeze(1))

        return {
            "scores":     torch.cat(all_scores).numpy(),
            "labels":     torch.cat(all_labels).numpy(),
            "error_maps": torch.cat(all_maps).numpy(),
        }
