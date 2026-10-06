"""Compact U-Net for 2D (k=0) and 2.5D (k>0 adjacent slices as channels) segmentation.

Differences from the seniors'/teammate's UNet:
* configurable depth and width, so the deployable model stays < 15 MB for in-browser ONNX;
* spatial dropout in the decoder for Monte-Carlo-dropout uncertainty;
* residual double-conv blocks (faster convergence on few labelled frames);
* ONNX-friendly (no dynamic padding: inputs are padded to a multiple of 2**depth outside).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 3, base: int = 16, depth: int = 5,
                 dropout: float = 0.1, max_ch: int = 256):
        super().__init__()
        self.depth = depth
        chs = [min(base * 2 ** i, max_ch) for i in range(depth + 1)]
        self.stem = ResBlock(in_channels, chs[0])
        self.down = nn.ModuleList([ResBlock(chs[i], chs[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([
            ResBlock(chs[i + 1] + chs[i], chs[i], dropout=dropout if i >= 1 else 0.0)
            for i in reversed(range(depth))
        ])
        self.head = nn.Conv2d(chs[0], num_classes, 1)
        self.config = dict(in_channels=in_channels, num_classes=num_classes, base=base, depth=depth,
                           dropout=dropout, max_ch=max_ch)

    def forward(self, x):
        skips = [self.stem(x)]
        for blk in self.down:
            skips.append(blk(F.max_pool2d(skips[-1], 2)))
        y = skips.pop()
        for blk in self.up:
            s = skips.pop()
            y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
            y = blk(torch.cat([y, s], 1))
        return self.head(y)

    @property
    def multiple(self) -> int:
        return 2 ** self.depth


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path, device: str = "cpu"):
    """Rebuild any zoo architecture from a checkpoint (no pretrained download) and load its weights."""
    from .zoo import build_model
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model({**ckpt["config"], "pretrained": False}).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
