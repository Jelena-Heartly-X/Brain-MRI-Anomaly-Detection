"""
models/encoder.py — CNN encoder with Squeeze-and-Excitation (SE) attention blocks.

Architecture:
  Input (1, H, W)
  → ConvBlock(64) → SEBlock → pool  → (64, H/2, W/2)
  → ConvBlock(128) → SEBlock → pool → (128, H/4, W/4)
  → ConvBlock(256) → SEBlock → pool → (256, H/8, W/8)
  → ConvBlock(512) → SEBlock        → (512, H/8, W/8)  [bottleneck, no pool]

Skip connections at each scale are returned for the decoder.
Optionally adds a self-attention layer at the bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU.
    Residual connection when in_ch == out_ch.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.drop  = nn.Dropout2d(dropout)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.bn_s  = nn.BatchNorm2d(out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.bn_s(self.skip(x))
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return F.relu(x + residual, inplace=True)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention block (Hu et al., 2018).
    Recalibrates channel-wise feature responses adaptively.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SpatialSelfAttention(nn.Module):
    """
    Lightweight spatial self-attention for the bottleneck.
    Uses Q/K/V projections with scaled dot-product attention.
    Operates on flattened spatial positions.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)  # each (B, heads, head_dim, HW)
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, H, W)
        return self.norm(self.proj(out)) + x


# ─── Encoder ──────────────────────────────────────────────────────────────────

class ConvMAEEncoder(nn.Module):
    """
    Hierarchical CNN encoder for Conv-MAE.
    Returns latent features AND skip connection tensors for the decoder.
    """

    def __init__(self, config):
        super().__init__()
        in_ch = 1  # Grayscale MRI
        channels = config.encoder_channels   # e.g. [64, 128, 256, 512]
        dropout = config.dropout
        use_se = config.use_se_block
        use_attn = config.use_self_attention

        self.stages = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        c = in_ch
        for i, out_c in enumerate(channels):
            self.stages.append(ConvBlock(c, out_c, dropout))
            self.se_blocks.append(SEBlock(out_c, config.se_reduction) if use_se else nn.Identity())
            # Last stage: no pooling (bottleneck)
            if i < len(channels) - 1:
                self.pools.append(nn.MaxPool2d(2, 2))
            else:
                self.pools.append(nn.Identity())
            c = out_c

        # Optional bottleneck self-attention
        self.bottleneck_attn = (
            SpatialSelfAttention(channels[-1]) if use_attn else nn.Identity()
        )

        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            latent:  (B, C_last, H', W') — bottleneck features
            skips:   list of (B, C_i, H_i, W_i) tensors at each scale
        """
        skips = []
        for stage, se, pool in zip(self.stages, self.se_blocks, self.pools):
            x = stage(x)
            x = se(x)
            skips.append(x)  # store before pooling
            x = pool(x)
        x = self.bottleneck_attn(x)
        return x, skips


# ─── 3D Encoder variant ───────────────────────────────────────────────────────

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.drop  = nn.Dropout3d(dropout)
        self.skip  = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.bn_s  = nn.BatchNorm3d(out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.bn_s(self.skip(x))
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return F.relu(x + residual, inplace=True)


class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        return x * self.fc(w).view(b, c, 1, 1, 1)


class ConvMAEEncoder3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config.encoder_channels
        dropout = config.dropout
        self.stages = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = 1
        for i, out_c in enumerate(channels):
            self.stages.append(ConvBlock3D(c, out_c, dropout))
            self.se_blocks.append(SEBlock3D(out_c, config.se_reduction) if config.use_se_block else nn.Identity())
            self.pools.append(nn.MaxPool3d(2) if i < len(channels) - 1 else nn.Identity())
            c = out_c
        self.out_channels = channels[-1]

    def forward(self, x):
        skips = []
        for stage, se, pool in zip(self.stages, self.se_blocks, self.pools):
            x = stage(x)
            x = se(x)
            skips.append(x)
            x = pool(x)
        return x, skips
