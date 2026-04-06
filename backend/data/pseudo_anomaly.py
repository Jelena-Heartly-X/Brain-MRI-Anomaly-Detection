"""
data/pseudo_anomaly.py — Synthetic anomaly generation for training the anomaly classifier.

Generates realistic-looking pseudo-anomalies on normal MRI slices by:
  1. Random bounding box + Gaussian noise injection
  2. Intensity distortion within a region
  3. Texture swap from a different slice/scan
  4. CutPaste-style patch transplant

These pseudo-anomalies are used to train the CNN classifier on error maps.
"""

import numpy as np
import torch
from typing import Tuple, Optional, List


class PseudoAnomalyGenerator:
    """
    Applies random synthetic anomalies to normal MRI slices.
    Returns the modified image, a binary anomaly mask, and an anomaly label.
    """

    def __init__(self, config):
        self.config = config
        self.anomaly_types = config.pseudo_anomaly_types

    def __call__(self, image: torch.Tensor, apply: bool = True
                 ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            image: (1, H, W) or (1, D, H, W) normal MRI tensor in [0, 1]
            apply: if False, return image unchanged with label=0
        Returns:
            (augmented_image, anomaly_mask, label)
        """
        if not apply:
            mask = torch.zeros_like(image)
            return image, mask, 0

        img_np = image.squeeze(0).numpy()  # (H, W)
        atype = np.random.choice(self.anomaly_types)

        if atype == "bbox_noise":
            img_np, mask_np = self._bbox_noise(img_np)
        elif atype == "intensity_distort":
            img_np, mask_np = self._intensity_distort(img_np)
        elif atype == "texture_swap":
            img_np, mask_np = self._texture_swap(img_np)
        else:
            img_np, mask_np = self._bbox_noise(img_np)

        aug_tensor = torch.from_numpy(img_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        return aug_tensor, mask_tensor, 1

    def _random_bbox(self, h: int, w: int,
                     min_frac: float = 0.05, max_frac: float = 0.25
                     ) -> Tuple[int, int, int, int]:
        """Return (y1, x1, y2, x2) for a random box within the image."""
        bh = np.random.randint(int(h * min_frac), int(h * max_frac) + 1)
        bw = np.random.randint(int(w * min_frac), int(w * max_frac) + 1)
        y1 = np.random.randint(0, h - bh)
        x1 = np.random.randint(0, w - bw)
        return y1, x1, y1 + bh, x1 + bw

    def _bbox_noise(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inject Gaussian noise inside a random bounding box."""
        h, w = img.shape
        y1, x1, y2, x2 = self._random_bbox(h, w)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        img = img.copy()
        noise_level = np.random.uniform(0.1, 0.4)
        img[y1:y2, x1:x2] = np.clip(
            img[y1:y2, x1:x2] + np.random.randn(y2-y1, x2-x1).astype(np.float32) * noise_level,
            0.0, 1.0
        )
        return img, mask

    def _intensity_distort(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply intensity inversion or strong shift inside a random elliptical region."""
        h, w = img.shape
        mask = np.zeros((h, w), dtype=np.float32)
        # Elliptical region
        cy, cx = np.random.randint(h // 4, 3 * h // 4), np.random.randint(w // 4, 3 * w // 4)
        ry = np.random.randint(int(h * 0.05), int(h * 0.15))
        rx = np.random.randint(int(w * 0.05), int(w * 0.15))
        yy, xx = np.ogrid[:h, :w]
        ellipse = ((yy - cy)**2 / ry**2 + (xx - cx)**2 / rx**2) <= 1
        mask[ellipse] = 1.0

        img = img.copy()
        distort_type = np.random.choice(["invert", "brighten", "darken", "gamma"])
        if distort_type == "invert":
            img[ellipse] = 1.0 - img[ellipse]
        elif distort_type == "brighten":
            img[ellipse] = np.clip(img[ellipse] + np.random.uniform(0.2, 0.5), 0, 1)
        elif distort_type == "darken":
            img[ellipse] = np.clip(img[ellipse] - np.random.uniform(0.2, 0.5), 0, 1)
        elif distort_type == "gamma":
            gamma = np.random.choice([0.3, 0.4, 2.5, 3.0])
            img[ellipse] = np.power(img[ellipse], gamma)
        return img, mask

    def _texture_swap(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transplant a patch from a random location to another (CutPaste-style)."""
        h, w = img.shape
        # Source patch
        y1s, x1s, y2s, x2s = self._random_bbox(h, w, 0.05, 0.2)
        ph, pw = y2s - y1s, x2s - x1s
        patch = img[y1s:y2s, x1s:x2s].copy()

        # Destination (non-overlapping)
        for _ in range(20):
            y1d = np.random.randint(0, h - ph)
            x1d = np.random.randint(0, w - pw)
            if abs(y1d - y1s) > ph or abs(x1d - x1s) > pw:
                break

        mask = np.zeros((h, w), dtype=np.float32)
        mask[y1d:y1d+ph, x1d:x1d+pw] = 1.0
        img = img.copy()
        # Blend with random alpha
        alpha = np.random.uniform(0.6, 1.0)
        img[y1d:y1d+ph, x1d:x1d+pw] = (
            alpha * patch + (1 - alpha) * img[y1d:y1d+ph, x1d:x1d+pw]
        )
        return img, mask


def augment_batch_with_pseudo_anomalies(
    images: torch.Tensor,
    config,
    fraction: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply pseudo-anomalies to `fraction` of samples in a batch.

    Args:
        images: (B, 1, H, W)
        config: Config object
        fraction: fraction of batch to corrupt

    Returns:
        (augmented_images, anomaly_masks, labels)  all (B, ...)
    """
    generator = PseudoAnomalyGenerator(config)
    B = images.shape[0]
    aug_images = images.clone()
    masks = torch.zeros_like(images)
    labels = torch.zeros(B, dtype=torch.long)

    n_corrupt = int(B * fraction)
    indices = np.random.choice(B, n_corrupt, replace=False)

    for i in indices:
        aug, mask, label = generator(images[i].cpu(), apply=True)
        aug_images[i] = aug
        masks[i] = mask
        labels[i] = label

    return aug_images, masks, labels
