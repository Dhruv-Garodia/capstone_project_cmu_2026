"""A small 2D U-Net whose input channels are neighbouring slices (2.5D)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """in_channels = 2k+1 slices; output = 2 logits (solid, pore). Input sides must be multiples of 2**depth."""

    def __init__(self, in_channels, base=32, depth=4):
        super().__init__()
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList(_block(cin, cout) for cin, cout in zip([in_channels] + widths[:-1], widths))
        self.up = nn.ModuleList(_block(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(base, 2, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        skips.pop()
        for block in self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skips.pop()], 1))
        return self.head(x)

    @property
    def stride(self):
        return 2 ** (len(self.down) - 1)
