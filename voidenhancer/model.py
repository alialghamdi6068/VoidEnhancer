"""VoidEnhancer neural network.

This is an original small super-resolution architecture implemented from scratch.
It is intentionally compact for the first training milestone; it is not a
pretrained model or a wrapper around an existing upscaler.
"""

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.body(x)


class VoidEnhancer(nn.Module):
    """A from-scratch residual CNN for 2x/4x super-resolution.

    The network predicts an RGB residual after bicubic upsampling. This first
    version is designed for image/frame training. Temporal video modules will
    be added after the single-frame baseline is validated.
    """

    def __init__(self, scale: int = 4, channels: int = 64, blocks: int = 8):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")

        self.scale = scale
        self.head = nn.Sequential(
            nn.Conv2d(3, channels, 5, padding=2),
            nn.PReLU(channels),
        )
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.trunk = nn.Conv2d(channels, channels, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(channels),
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        features = self.head(x)
        residual = self.trunk(self.body(features))
        features = features + residual
        enhanced = self.tail(self.up(features))
        return torch.clamp(base + enhanced, 0.0, 1.0)
