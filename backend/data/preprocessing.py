"""
data/preprocessing.py — Research-grade MRI preprocessing pipeline.

Implements:
  - NIfTI → slice/volume extraction
  - N4ITK bias field correction (via SimpleITK)
  - Skull stripping (via antspyx or simple intensity masking)
  - Empty / non-brain slice filtering
  - CLAHE histogram equalization
  - Z-score and min-max normalization
  - Gaussian / median noise reduction
  - Intensity standardization across scans
  - Augmentation: rotation, flip, elastic deformation, intensity shift, random crop
"""

import numpy as np
import SimpleITK as sitk
import nibabel as nib
from skimage import exposure, filters, transform, morphology
from scipy.ndimage import gaussian_filter, median_filter, map_coordinates
import torch
from torch import Tensor
import torchvision.transforms.functional as TF
from typing import Optional, Tuple, Dict
import warnings

warnings.filterwarnings("ignore")


# ─── N4ITK Bias Field Correction ─────────────────────────────────────────────

def n4itk_correction(sitk_image: sitk.Image, n_iterations: int = 50) -> sitk.Image:
    """Apply N4ITK bias field correction to a SimpleITK image."""
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([n_iterations] * 4)
    try:
        mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
        return corrector.Execute(sitk_image, mask)
    except Exception:
        return corrector.Execute(sitk_image)


# ─── Skull Stripping ──────────────────────────────────────────────────────────

def simple_skull_strip(volume: np.ndarray, threshold_pct: float = 15.0) -> np.ndarray:
    """
    Lightweight intensity-based skull strip.
    For research quality, replace with HD-BET or SynthStrip.
    """
    threshold = np.percentile(volume[volume > 0], threshold_pct)
    brain_mask = volume > threshold
    # Morphological cleanup
    brain_mask = morphology.binary_fill_holes(brain_mask)
    for _ in range(3):
        brain_mask = morphology.binary_erosion(brain_mask)
    for _ in range(3):
        brain_mask = morphology.binary_dilation(brain_mask)
    return volume * brain_mask


# ─── Volume Loading ────────────────────────────────────────────────────────────

def load_nifti_volume(path: str, apply_n4: bool = True, skull_strip: bool = False
                      ) -> np.ndarray:
    """
    Load a NIfTI file, apply N4ITK correction and optional skull stripping.
    Returns float32 volume with shape (D, H, W).
    """
    sitk_img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if apply_n4:
        sitk_img = n4itk_correction(sitk_img)
    volume = sitk.GetArrayFromImage(sitk_img).astype(np.float32)  # (D, H, W)
    if skull_strip:
        volume = simple_skull_strip(volume)
    return volume


# ─── Normalization ────────────────────────────────────────────────────────────

def zscore_normalize(volume: np.ndarray, brain_mask: Optional[np.ndarray] = None
                     ) -> np.ndarray:
    """Z-score normalize using brain voxels only."""
    if brain_mask is not None:
        voxels = volume[brain_mask > 0]
    else:
        voxels = volume[volume > 0]
    mean, std = voxels.mean(), voxels.std()
    std = std if std > 1e-6 else 1.0
    return (volume - mean) / std


def minmax_normalize(volume: np.ndarray, vmin: float = 0.0, vmax: float = 1.0
                     ) -> np.ndarray:
    lo, hi = volume.min(), volume.max()
    if hi - lo < 1e-6:
        return np.zeros_like(volume)
    return (volume - lo) / (hi - lo) * (vmax - vmin) + vmin


def percentile_clip(volume: np.ndarray, low: float = 0.5, high: float = 99.5
                    ) -> np.ndarray:
    """Clip to robust percentile range before normalization."""
    plo, phi = np.percentile(volume, low), np.percentile(volume, high)
    return np.clip(volume, plo, phi)


# ─── Image Enhancement ────────────────────────────────────────────────────────

def apply_clahe(slice_2d: np.ndarray, clip_limit: float = 0.03,
                nbins: int = 256) -> np.ndarray:
    """Apply CLAHE (contrast limited adaptive histogram equalization)."""
    img = minmax_normalize(slice_2d)
    return exposure.equalize_adapthist(img, clip_limit=clip_limit, nbins=nbins)


def reduce_noise(slice_2d: np.ndarray, method: str = "gaussian",
                 sigma: float = 0.5) -> np.ndarray:
    if method == "gaussian":
        return gaussian_filter(slice_2d, sigma=sigma)
    elif method == "median":
        return median_filter(slice_2d, size=3)
    return slice_2d


# ─── Slice Extraction ─────────────────────────────────────────────────────────

def is_brain_slice(slice_2d: np.ndarray, min_fraction: float = 0.02) -> bool:
    """Return True if the slice contains sufficient brain tissue."""
    brain_pixels = np.sum(slice_2d > slice_2d.max() * 0.05)
    return brain_pixels / slice_2d.size > min_fraction


def extract_slices(volume: np.ndarray, target_size: int = 224,
                   min_brain_fraction: float = 0.02,
                   apply_clahe_flag: bool = True,
                   noise_method: str = "gaussian") -> list:
    """
    Extract 2D slices from a volume, applying preprocessing and filtering.
    Returns list of preprocessed (H, W) float32 arrays.
    """
    slices = []
    for i in range(volume.shape[0]):
        s = volume[i]
        if not is_brain_slice(s, min_brain_fraction):
            continue
            
        # Crop to brain bounding box to prevent MAE from masking only empty background
        rows = np.any(s > 0, axis=1)
        cols = np.any(s > 0, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            pad = 10
            rmin = max(0, rmin - pad)
            rmax = min(s.shape[0], rmax + pad)
            cmin = max(0, cmin - pad)
            cmax = min(s.shape[1], cmax + pad)
            s = s[rmin:rmax, cmin:cmax]

        # Resize
        s = transform.resize(s, (target_size, target_size),
                              order=3, anti_aliasing=True, preserve_range=True)
        # Denoise
        s = reduce_noise(s, method=noise_method)
        # CLAHE
        if apply_clahe_flag:
            s = apply_clahe(s)
        else:
            s = minmax_normalize(s)
        slices.append(s.astype(np.float32))
    return slices


# ─── 3D Volume Preprocessing ──────────────────────────────────────────────────

def preprocess_volume_3d(volume: np.ndarray,
                          target_size: Tuple[int, int, int] = (96, 112, 96),
                          apply_clahe_flag: bool = False) -> np.ndarray:
    """Resize, normalize, and optionally enhance a 3D volume."""
    # Percentile clip outliers
    volume = percentile_clip(volume)
    volume = zscore_normalize(volume)
    # Resize each axis with zoom
    from scipy.ndimage import zoom
    factors = [t / s for t, s in zip(target_size, volume.shape)]
    volume = zoom(volume, factors, order=3)
    volume = minmax_normalize(volume)
    return volume.astype(np.float32)


# ─── Augmentation ─────────────────────────────────────────────────────────────

def elastic_deformation(image: np.ndarray, alpha: float = 100.0,
                         sigma: float = 10.0, seed: int = None) -> np.ndarray:
    """Apply elastic deformation to a 2D image."""
    rng = np.random.RandomState(seed)
    shape = image.shape
    dx = gaussian_filter(rng.rand(*shape) * 2 - 1, sigma) * alpha
    dy = gaussian_filter(rng.rand(*shape) * 2 - 1, sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords = [np.clip(y + dy, 0, shape[0]-1), np.clip(x + dx, 0, shape[1]-1)]
    return map_coordinates(image, coords, order=1).astype(image.dtype)


class MRIAugmentation:
    """
    Stochastic augmentation pipeline for 2D MRI slices.
    All operations preserve float32 output in [0, 1].
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, image: np.ndarray) -> np.ndarray:
        cfg = self.config

        # Random horizontal flip
        if np.random.rand() < cfg.aug_flip_prob:
            image = np.fliplr(image).copy()

        # Random vertical flip (less common in brain MRI but valid augmentation)
        if np.random.rand() < 0.2:
            image = np.flipud(image).copy()

        # Random rotation
        angle = np.random.uniform(-cfg.aug_rotation_deg, cfg.aug_rotation_deg)
        image = transform.rotate(image, angle, mode="reflect", preserve_range=True)

        # Elastic deformation
        if np.random.rand() < 0.3:
            image = elastic_deformation(
                image, alpha=cfg.aug_elastic_alpha, sigma=cfg.aug_elastic_sigma
            )

        # Random crop + resize back
        scale = np.random.uniform(*cfg.aug_random_crop_scale)
        if scale < 1.0:
            h, w = image.shape
            ch, cw = int(h * scale), int(w * scale)
            top = np.random.randint(0, h - ch + 1)
            left = np.random.randint(0, w - cw + 1)
            image = image[top:top+ch, left:left+cw]
            image = transform.resize(image, (h, w), order=3,
                                     anti_aliasing=True, preserve_range=True)

        # Random intensity shift
        shift = np.random.uniform(-cfg.aug_intensity_shift, cfg.aug_intensity_shift)
        image = np.clip(image + shift, 0.0, 1.0)

        # Random gamma correction
        if np.random.rand() < 0.3:
            gamma = np.random.uniform(0.8, 1.2)
            image = np.power(np.clip(image, 0, 1), gamma)

        return image.astype(np.float32)


# ─── Intensity Standardization Across Scans ───────────────────────────────────

class IntensityStandardizer:
    """
    Learns a histogram matching reference from training data.
    Applies standardization to bring new scans into a common intensity space.
    Based on Nyul et al. (2000) piecewise linear histogram matching.
    """

    def __init__(self, percentiles: list = None):
        self.percentiles = percentiles or [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
        self.standard_scale: Optional[np.ndarray] = None

    def fit(self, volumes: list) -> "IntensityStandardizer":
        """Compute the standard intensity scale from a list of volumes."""
        all_percentiles = []
        for vol in volumes:
            brain = vol[vol > vol.max() * 0.05]
            all_percentiles.append(np.percentile(brain, self.percentiles))
        self.standard_scale = np.mean(all_percentiles, axis=0)
        return self

    def transform(self, volume: np.ndarray) -> np.ndarray:
        """Map a volume's intensities to the standard scale."""
        if self.standard_scale is None:
            return volume
        brain_mask = volume > volume.max() * 0.05
        brain = volume[brain_mask]
        source = np.percentile(brain, self.percentiles)
        # Piecewise linear mapping
        result = np.interp(volume.ravel(), source, self.standard_scale)
        return result.reshape(volume.shape).astype(np.float32)
