"""
utils/config.py — Centralized configuration for Conv-MAE MRI anomaly detection.
All hyperparameters, paths, and flags live here. Override via subclassing or
direct attribute assignment before passing to Trainer / Evaluator.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Config:
    # ─── Paths ────────────────────────────────────────────────────────────────
    ixi_dir: str = "/content/drive/MyDrive/data/IXI"          # Normal MRI (train)
    brats_dir: str = "/content/drive/MyDrive/data/BraTS2020"   # Tumor MRI (test)
    figshare_dir: Optional[str] = None                         # Optional extra test set
    checkpoint_dir: str = "/content/drive/MyDrive/checkpoints/conv_mae"
    output_dir: str = "/content/drive/MyDrive/outputs/conv_mae"

    # ─── Data ─────────────────────────────────────────────────────────────────
    image_size: int = 224                  # 2D slice size (H = W)
    volume_size: Tuple[int, int, int] = (96, 112, 96)  # 3D input shape (D, H, W)
    modality: str = "T1"                   # MRI modality: T1 | T2 | FLAIR
    use_3d: bool = False                   # Toggle 2D vs 3D pipeline
    min_brain_fraction: float = 0.02       # Drop slices with < 2% brain voxels
    val_split: float = 0.1                 # Fraction of IXI used for validation
    num_workers: int = 4

    # ─── Preprocessing ────────────────────────────────────────────────────────
    apply_n4itk: bool = True               # Bias field correction
    apply_skull_strip: bool = False        # Skull stripping (requires HD-BET or antspyx)
    apply_clahe: bool = True               # CLAHE histogram equalization
    apply_zscore: bool = True              # Z-score normalization
    apply_minmax: bool = False             # Min-max normalization (alternative)
    noise_reduction: str = "gaussian"      # "gaussian" | "median" | "none"

    # ─── Augmentation ─────────────────────────────────────────────────────────
    augment: bool = True
    aug_flip_prob: float = 0.5
    aug_rotation_deg: float = 15.0
    aug_elastic_alpha: float = 100.0
    aug_elastic_sigma: float = 10.0
    aug_intensity_shift: float = 0.1
    aug_random_crop_scale: Tuple[float, float] = (0.85, 1.0)

    # ─── Masking ──────────────────────────────────────────────────────────────
    mask_ratio: float = 0.75               # Fraction of patches masked
    patch_size: int = 16                   # Patch size for masking grid
    block_masking: bool = True             # Block-wise vs random masking

    # ─── Model Architecture ───────────────────────────────────────────────────
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    use_se_block: bool = True              # Squeeze-and-Excitation blocks in encoder
    se_reduction: int = 16                 # SE reduction ratio
    use_self_attention: bool = False       # Optional self-attention bottleneck
    dropout: float = 0.1

    # ─── Loss ─────────────────────────────────────────────────────────────────
    loss_mse_weight: float = 0.5
    loss_ssim_weight: float = 0.5
    loss_perceptual_weight: float = 0.0    # Set > 0 to enable perceptual loss
    loss_l1_weight: float = 0.0           # Optional L1 component

    # ─── Training ─────────────────────────────────────────────────────────────
    optimizer: str = "adamw"              # adam | adamw | rmsprop | sgd | lookahead
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 100
    grad_clip: float = 1.0
    mixed_precision: bool = True           # FP16 training via torch.cuda.amp
    scheduler: str = "cosine"             # cosine | step | plateau
    lr_warmup_epochs: int = 5
    early_stopping_patience: int = 15
    log_every: int = 10                   # Log every N batches

    # ─── Pseudo-Anomaly (for classifier training) ─────────────────────────────
    pseudo_anomaly_fraction: float = 0.5  # 50% of training samples get pseudo-anomaly
    pseudo_anomaly_types: List[str] = field(
        default_factory=lambda: ["bbox_noise", "intensity_distort", "texture_swap"]
    )

    # ─── Anomaly Scoring ──────────────────────────────────────────────────────
    score_mse_weight: float = 0.4
    score_ssim_weight: float = 0.4
    score_l1_weight: float = 0.2
    multiscale_levels: int = 3            # Number of resolution levels for multi-scale score
    gaussian_smooth_sigma: float = 3.0   # Heatmap smoothing sigma

    # ─── Evaluation ───────────────────────────────────────────────────────────
    threshold_method: str = "otsu"        # otsu | percentile | fixed
    threshold_percentile: float = 95.0   # Used if threshold_method == "percentile"
    threshold_fixed: float = 0.5          # Used if threshold_method == "fixed"

    # ─── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda"                  # "cuda" | "cpu"

    def __post_init__(self):
        import os, torch, random, numpy as np
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        # Seed everything
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
