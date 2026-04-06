"""
evaluation/heatmap.py — Anomaly localization heatmap generation.

Implements:
  - Gaussian-smoothed pixel error maps
  - Normalized JET colormap overlay on original MRI
  - Adaptive Otsu / percentile thresholding
  - Binary segmentation mask generation
  - Morphological post-processing
  - Multi-panel visualization (original, recon, error, heatmap, mask)
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter, binary_fill_holes, label as cc_label
from skimage import morphology, filters
from typing import Optional, Tuple, Dict


# ─── Heatmap Generation ───────────────────────────────────────────────────────

def smooth_error_map(error_map: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Apply Gaussian smoothing to reduce noise in the error map."""
    return gaussian_filter(error_map, sigma=sigma)


def normalize_map(error_map: np.ndarray,
                  percentile_clip: float = 99.0) -> np.ndarray:
    """
    Normalize error map to [0, 1].
    Clips at `percentile_clip` to suppress extreme outliers.
    """
    clip_val = np.percentile(error_map, percentile_clip)
    err = np.clip(error_map, 0, clip_val)
    lo, hi = err.min(), err.max()
    if hi - lo < 1e-8:
        return np.zeros_like(err)
    return (err - lo) / (hi - lo)


def apply_jet_colormap(normalized_map: np.ndarray) -> np.ndarray:
    """
    Apply JET colormap to a normalized [0, 1] map.
    Returns (H, W, 3) RGB array in [0, 255] uint8.
    """
    jet = cm.get_cmap("jet")
    rgb = jet(normalized_map)[:, :, :3]  # drop alpha
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(mri_slice: np.ndarray, heatmap_rgb: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """
    Blend heatmap over grayscale MRI slice.
    mri_slice:   (H, W) float in [0, 1]
    heatmap_rgb: (H, W, 3) uint8
    Returns (H, W, 3) uint8 blended image.
    """
    mri_rgb = np.stack([mri_slice * 255] * 3, axis=-1).astype(np.uint8)
    blended = ((1 - alpha) * mri_rgb + alpha * heatmap_rgb).astype(np.uint8)
    return blended


# ─── Thresholding ─────────────────────────────────────────────────────────────

def compute_threshold(error_map: np.ndarray, method: str = "otsu",
                      percentile: float = 95.0,
                      fixed: float = 0.5) -> float:
    """
    Compute binary threshold for anomaly mask.

    Methods:
      - 'otsu':       Otsu's method on the error map distribution
      - 'percentile': Fixed percentile of error values
      - 'fixed':      Fixed scalar threshold
    """
    if method == "otsu":
        try:
            return filters.threshold_otsu(error_map)
        except Exception:
            return np.percentile(error_map, 90)
    elif method == "percentile":
        return np.percentile(error_map, percentile)
    elif method == "fixed":
        return fixed
    return np.percentile(error_map, 90)


def generate_binary_mask(error_map: np.ndarray,
                          threshold: float,
                          min_area: int = 50) -> np.ndarray:
    """
    Generate binary anomaly mask from thresholded error map.
    Applies morphological cleanup to remove noise.

    Args:
        error_map:  (H, W) float normalized error
        threshold:  scalar threshold
        min_area:   minimum connected component area to keep

    Returns:
        binary mask (H, W) uint8
    """
    mask = (error_map > threshold).astype(np.uint8)
    # Morphological operations
    mask = binary_fill_holes(mask).astype(np.uint8)
    mask = morphology.binary_opening(mask, morphology.disk(2)).astype(np.uint8)
    mask = morphology.binary_closing(mask, morphology.disk(3)).astype(np.uint8)

    # Remove small connected components
    labeled, n_components = cc_label(mask)
    for comp_id in range(1, n_components + 1):
        if (labeled == comp_id).sum() < min_area:
            mask[labeled == comp_id] = 0

    return mask


# ─── Full Heatmap Pipeline ────────────────────────────────────────────────────

class AnomalyHeatmapGenerator:
    """
    Complete heatmap generation pipeline for a single MRI slice.
    Takes raw error tensors → produces annotated visualizations.
    """

    def __init__(self, config):
        self.config = config

    def process(
        self,
        original: np.ndarray,       # (H, W) float [0, 1]
        reconstruction: np.ndarray, # (H, W) float [0, 1]
        error_map: np.ndarray,      # (H, W) float
        gt_mask: Optional[np.ndarray] = None,  # (H, W) uint8 ground-truth
    ) -> Dict:
        """
        Full pipeline: smooth → normalize → colorize → threshold → mask.

        Returns dict with:
          heatmap_rgb:     (H, W, 3) JET-colored error
          overlay:         (H, W, 3) heatmap on MRI
          binary_mask:     (H, W) uint8 predicted anomaly mask
          normalized_error:(H, W) float normalized map
          threshold:       scalar threshold used
        """
        # 1. Smooth
        smoothed = smooth_error_map(error_map, sigma=self.config.gaussian_smooth_sigma)

        # 2. Normalize
        normed = normalize_map(smoothed)

        # 3. Colorize
        heatmap_rgb = apply_jet_colormap(normed)

        # 4. Overlay
        overlay = overlay_heatmap(original, heatmap_rgb, alpha=0.45)

        # 5. Threshold + binary mask
        threshold = compute_threshold(
            normed,
            method=self.config.threshold_method,
            percentile=self.config.threshold_percentile,
            fixed=self.config.threshold_fixed,
        )
        binary_mask = generate_binary_mask(normed, threshold)

        return {
            "heatmap_rgb":      heatmap_rgb,
            "overlay":          overlay,
            "binary_mask":      binary_mask,
            "normalized_error": normed,
            "threshold":        threshold,
        }

    def visualize(
        self,
        original: np.ndarray,
        reconstruction: np.ndarray,
        error_map: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        title: str = "",
    ) -> plt.Figure:
        """
        Generate a multi-panel figure:
        [Original | Reconstruction | Error Map | Heatmap Overlay | Pred Mask | GT Mask*]
        *GT mask only shown if provided.
        """
        result = self.process(original, reconstruction, error_map, gt_mask)

        n_panels = 6 if gt_mask is not None else 5
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

        panels = [
            (original, "gray", "Original MRI"),
            (reconstruction, "gray", "Reconstruction"),
            (result["normalized_error"], "hot", "Error Map"),
            (result["overlay"], None, "Heatmap Overlay"),
            (result["binary_mask"], "gray", "Predicted Mask"),
        ]
        if gt_mask is not None:
            panels.append((gt_mask, "gray", "Ground Truth"))

        for ax, (img, cmap, label) in zip(axes, panels):
            if cmap:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if img.max() <= 1 else None)
            else:
                ax.imshow(img)
            ax.set_title(label, fontsize=10)
            ax.axis("off")

        plt.suptitle(title, fontsize=12, y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fig
