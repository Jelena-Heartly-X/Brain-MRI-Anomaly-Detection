import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        merged_ch = in_ch + skip_ch
        self.conv = nn.Sequential(
            nn.Conv2d(merged_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ConvMAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        enc_ch = list(reversed(config.encoder_channels))
        dec_ch = config.decoder_channels
        dropout = config.dropout
        self.up_blocks = nn.ModuleList()
        self.aux_heads  = nn.ModuleList()
        in_ch = enc_ch[0]
        for i in range(3):
            skip_c = enc_ch[i + 1]
            out_c  = dec_ch[i]
            self.up_blocks.append(UpBlock(in_ch, skip_c, out_c, dropout))
            self.aux_heads.append(nn.Conv2d(out_c, 1, 1))
            in_ch = out_c
        self.final_head = nn.Sequential(
            nn.Conv2d(in_ch, dec_ch[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_ch[-1], 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        use_skips = list(reversed(skips[:-1]))
        aux_recons = []
        for up_block, aux_head, skip in zip(self.up_blocks, self.aux_heads, use_skips):
            x = up_block(x, skip)
            aux_recons.append(torch.sigmoid(aux_head(x)))
        recon = self.final_head(x)
        return recon, aux_recons


class UpBlock3D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ConvMAEDecoder3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        enc_ch = list(reversed(config.encoder_channels))
        dec_ch = config.decoder_channels
        dropout = config.dropout
        self.up_blocks = nn.ModuleList()
        self.aux_heads  = nn.ModuleList()
        in_ch = enc_ch[0]
        for i in range(3):
            skip_c = enc_ch[i + 1]
            out_c  = dec_ch[i]
            self.up_blocks.append(UpBlock3D(in_ch, skip_c, out_c, dropout))
            self.aux_heads.append(nn.Conv3d(out_c, 1, 1))
            in_ch = out_c
        self.final_head = nn.Sequential(
            nn.Conv3d(in_ch, dec_ch[-1], 3, padding=1, bias=False),
            nn.BatchNorm3d(dec_ch[-1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(dec_ch[-1], 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, skips):
        use_skips = list(reversed(skips[:-1]))
        aux_recons = []
        for up_block, aux_head, skip in zip(self.up_blocks, self.aux_heads, use_skips):
            x = up_block(x, skip)
            aux_recons.append(torch.sigmoid(aux_head(x)))
        recon = self.final_head(x)
        return recon, aux_recons
