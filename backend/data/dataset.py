"""
data/dataset.py — Dataset loaders for IXI (normal) and BraTS 2020 (anomaly).

IXIDataset   : Loads normal brain MRI slices for self-supervised Conv-MAE training.
BraTSDataset : Loads tumor MRI slices + segmentation masks for evaluation.
FigshareDataset : Optional supplementary test set.

Both support 2D and 3D modes, configurable via Config.
"""

import os
import glob
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib

from data.preprocessing import (
    load_nifti_volume, extract_slices, preprocess_volume_3d,
    zscore_normalize, minmax_normalize, percentile_clip, MRIAugmentation
)
from utils.config import Config


# ─── IXI Dataset (Normal MRI — Training) ─────────────────────────────────────

class IXIDataset(Dataset):
    """
    Loads T1/T2/PD MRI from IXI dataset.
    Only uses normal subjects for self-supervised Conv-MAE training.

    Directory structure expected:
        ixi_dir/
          IXI001-Guys-0828-T1.nii.gz
          IXI002-HH-1530-T1.nii.gz
          ...
    """

    def __init__(self, config: Config, split: str = "train"):
        self.config = config
        self.split = split
        self.augment = MRIAugmentation(config) if (split == "train" and config.augment) else None

        # Gather all NIfTI files for specified modality
        pattern = os.path.join(config.ixi_dir, f"*-{config.modality}.nii.gz")
        all_files = sorted(glob.glob(pattern))
        all_files = all_files[:150]   # LIMIT DATASET (prevents RAM crash)
        if not all_files:
            # Fallback: search recursively
            all_files = sorted(Path(config.ixi_dir).rglob(f"*{config.modality}*.nii*"))
            all_files = [str(f) for f in all_files]

        if not all_files:
            raise FileNotFoundError(
                f"No NIfTI files found for modality {config.modality} in {config.ixi_dir}"
            )

        # Train/val split by subject (not by slice) to prevent data leakage
        random.seed(config.seed)
        random.shuffle(all_files)
        n_val = max(1, int(len(all_files) * config.val_split))
        if split == "train":
            self.files = all_files[n_val:]
        else:
            self.files = all_files[:n_val]

        # Pre-extract and cache slices (in 2D mode)
        if not config.use_3d:
            self.slices = self._extract_all_slices()
        else:
            self.volumes = self.files  # lazy load for 3D

    def _extract_all_slices(self) -> List[np.ndarray]:
        """Extract, preprocess, and cache all brain slices from all volumes."""
        all_slices = []
        for f in self.files:
            try:
                volume = load_nifti_volume(
                    f,
                    apply_n4=self.config.apply_n4itk,
                    skull_strip=self.config.apply_skull_strip
                )
                slices = extract_slices(
                    volume,
                    target_size=self.config.image_size,
                    min_brain_fraction=self.config.min_brain_fraction,
                    apply_clahe_flag=self.config.apply_clahe,
                    noise_method=self.config.noise_reduction
                )
                all_slices.extend(slices)
            except Exception as e:
                print(f"[IXI] Warning: failed to load {f}: {e}")
        print(f"[IXI] Loaded {len(all_slices)} slices from {len(self.files)} subjects ({self.split})")
        return all_slices

    def __len__(self) -> int:
        if self.config.use_3d:
            return len(self.volumes)
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.config.use_3d:
            return self._get_volume(idx)
        return self._get_slice(idx)

    def _get_slice(self, idx: int) -> Dict[str, torch.Tensor]:
        img = self.slices[idx].copy()
        if self.augment:
            img = self.augment(img)
        # Shape: (1, H, W) — single channel grayscale
        tensor = torch.from_numpy(img).unsqueeze(0)
        return {"image": tensor, "label": torch.tensor(0)}  # 0 = normal

    def _get_volume(self, idx: int) -> Dict[str, torch.Tensor]:
        volume = load_nifti_volume(
            self.volumes[idx],
            apply_n4=self.config.apply_n4itk
        )
        volume = preprocess_volume_3d(volume, self.config.volume_size)
        tensor = torch.from_numpy(volume).unsqueeze(0)  # (1, D, H, W)
        return {"image": tensor, "label": torch.tensor(0)}


# ─── BraTS 2020 Dataset (Anomaly — Test) ─────────────────────────────────────

class BraTSDataset(Dataset):
    """
    Loads multi-modal brain tumor MRI and segmentation masks from BraTS 2020.

    Directory structure expected:
        brats_dir/
          BraTS20_Training_001/
            BraTS20_Training_001_t1.nii.gz
            BraTS20_Training_001_t1ce.nii.gz
            BraTS20_Training_001_t2.nii.gz
            BraTS20_Training_001_flair.nii.gz
            BraTS20_Training_001_seg.nii.gz
          BraTS20_Training_002/
            ...
    """

    # BraTS modality filename suffixes
    MODALITY_MAP = {
        "T1": ["_t1.nii.gz", "-t1.nii.gz", "-T1.nii.gz"],
        "T1CE": "_t1ce.nii.gz",
        "T2": "_t2.nii.gz",
        "FLAIR": "_flair.nii.gz",
    }

    def __init__(self, config: Config):
        self.config = config
        modality_suffix = self.MODALITY_MAP.get(config.modality, "_t1.nii.gz")

        subject_dirs = sorted([
            d for d in Path(config.brats_dir).iterdir() if d.is_dir()
        ])

        if not subject_dirs:
            raise FileNotFoundError(f"No BraTS subjects found in {config.brats_dir}")

        if not config.use_3d:
            self.samples = []  # (image_slice, mask_slice)
            self._load_slices(subject_dirs, modality_suffix)
        else:
            self.subjects = subject_dirs
            self.modality_suffix = modality_suffix

    def _load_slices(self, subject_dirs, modality_suffix):
        for subj_dir in subject_dirs:
            subj_name = subj_dir.name
            img_path = None
            for suffix in modality_suffix:
                p = subj_dir / (subj_name + suffix)
                if p.exists():
                    img_path = p
                    break
            seg_path = subj_dir / (subj_name + "_seg.nii.gz")

            if img_path is None:
                continue

            try:
                volume = load_nifti_volume(str(img_path), apply_n4=False)
                mask_vol = nib.load(str(seg_path)).get_fdata().astype(np.uint8) if seg_path.exists() else None

                from data.preprocessing import (
                    is_brain_slice, reduce_noise, apply_clahe,
                    minmax_normalize, transform
                )
                from skimage import transform as stf

                for i in range(volume.shape[0]):
                    sl = volume[i]
                    if not is_brain_slice(sl, self.config.min_brain_fraction):
                        continue

                    # Crop to brain bounding box identically for sl and mk
                    rows = np.any(sl > 0, axis=1)
                    cols = np.any(sl > 0, axis=0)
                    if rows.any() and cols.any():
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        pad = 10
                        rmin = max(0, rmin - pad)
                        rmax = max(rmin + 1, min(sl.shape[0], rmax + pad))
                        cmin = max(0, cmin - pad)
                        cmax = max(cmin + 1, min(sl.shape[1], cmax + pad))
                        sl = sl[rmin:rmax, cmin:cmax]
                        if mask_vol is not None:
                            mk = mask_vol[i][rmin:rmax, cmin:cmax]

                    sl = stf.resize(sl, (self.config.image_size, self.config.image_size),
                                    order=3, anti_aliasing=True, preserve_range=True)
                    sl = minmax_normalize(sl).astype(np.float32)

                    if mask_vol is not None:
                        mk = stf.resize(mk, (self.config.image_size, self.config.image_size),
                                        order=0, preserve_range=True).astype(np.uint8)
                        # BraTS labels: 1=necrosis, 2=edema, 4=enhancing → binary
                        mk = (mk > 0).astype(np.uint8)
                        has_anomaly = int(mk.sum() > 0)
                    else:
                        mk = np.zeros((self.config.image_size, self.config.image_size), dtype=np.uint8)
                        has_anomaly = 0

                    self.samples.append((sl, mk, has_anomaly))
            except Exception as e:
                print(f"[BraTS] Warning: failed {subj_dir}: {e}")

        print(f"[BraTS] Loaded {len(self.samples)} slices, "
              f"{sum(s[2] for s in self.samples)} with tumor")

    def __len__(self) -> int:
        if self.config.use_3d:
            return len(self.subjects)
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.config.use_3d:
            return self._get_volume_3d(idx)

        img, mask, label = self.samples[idx]
        return {
            "image": torch.from_numpy(img).unsqueeze(0),   # (1, H, W)
            "mask":  torch.from_numpy(mask).unsqueeze(0),  # (1, H, W)
            "label": torch.tensor(label),                   # 0 or 1
        }

    def _get_volume_3d(self, idx):
        subj_dir = self.subjects[idx]
        subj_name = subj_dir.name
        img_path = subj_dir / (subj_name + self.modality_suffix)
        seg_path = subj_dir / (subj_name + "_seg.nii.gz")
        volume = load_nifti_volume(str(img_path), apply_n4=False)
        volume = preprocess_volume_3d(volume, self.config.volume_size)
        seg = nib.load(str(seg_path)).get_fdata() if seg_path.exists() else np.zeros_like(volume)
        from scipy.ndimage import zoom
        seg = zoom(seg, [t/s for t, s in zip(self.config.volume_size, seg.shape)], order=0)
        seg = (seg > 0).astype(np.float32)
        return {
            "image": torch.from_numpy(volume).unsqueeze(0),
            "mask":  torch.from_numpy(seg).unsqueeze(0),
            "label": torch.tensor(int(seg.sum() > 0)),
        }


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def get_dataloaders(config: Config):
    """
    Returns train, val, and test DataLoaders.
    Train/val: IXI (normal only). Test: BraTS 2020.
    """
    train_ds = IXIDataset(config, split="train")
    val_ds   = IXIDataset(config, split="val")
    test_ds  = BraTSDataset(config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, val_loader, test_loader
