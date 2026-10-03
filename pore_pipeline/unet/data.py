"""Training patches and full-slice windows for the 2.5D U-Net.

Input  : (2k+1, H, W) float32 in [0, 1] — preprocessed slices z-k .. z+k (clamped at stack ends).
Target : (H, W) int64 — 1 pore, 0 solid, IGNORE for excluded / uncertain pixels.
Only the target slice needs a label; its neighbours are context.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .. import labels as L

IGNORE = 255


def parse_slices(text):
    """'0-90,100-117' -> [0..90, 100..117]"""
    zs = []
    for part in text.replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-")
            zs.extend(range(int(a), int(b) + 1))
        elif part:
            zs.append(int(part))
    return zs


def to_target(mask):
    target = np.full(mask.shape, IGNORE, np.int64)
    target[mask == L.PORE] = 1
    target[mask == L.SOLID] = 0
    return target


def window(frames, z, k):
    """(2k+1, H, W) uint8 view of slices z-k..z+k, repeating the end slices at the stack ends."""
    idx = np.clip(np.arange(z - k, z + k + 1), 0, len(frames) - 1)
    return frames[idx]


class PatchDataset(Dataset):
    """Random patches centred on labelled pixels of the given slices."""

    def __init__(self, frames, label_dir, zs, k, patch=256, length=4000, augment=True, seed=0):
        files = L.slice_files(label_dir)
        self.zs = [z for z in zs if z in files]
        self.frames, self.k, self.patch, self.length, self.augment = frames, k, patch, length, augment
        self.targets = {z: to_target(L.read_mask(label_dir, z)) for z in self.zs}
        self.rows = {z: np.flatnonzero((t != IGNORE).any(1)) for z, t in self.targets.items()}
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        rng, p = self.rng, self.patch
        z = self.zs[rng.integers(len(self.zs))]
        target = self.targets[z]
        h, w = target.shape
        cy = self.rows[z][rng.integers(len(self.rows[z]))]
        y0 = int(np.clip(cy - p // 2, 0, h - p))
        x0 = int(rng.integers(0, w - p + 1))
        x = window(self.frames, z, self.k)[:, y0:y0 + p, x0:x0 + p].astype(np.float32) / 255.0
        t = target[y0:y0 + p, x0:x0 + p]
        if self.augment:
            if rng.random() < 0.5:                       # left-right is the only symmetry of the imaging geometry
                x, t = x[:, :, ::-1], t[:, ::-1]
            gain, offset = rng.uniform(0.85, 1.15), rng.uniform(-0.06, 0.06)
            x = np.clip(x * gain + offset, 0, 1)         # tolerate brightness / contrast differences between stacks
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(t))


def pad_to(x, stride):
    """Reflect-pad a (..., H, W) tensor so both sides are multiples of `stride`; returns (padded, (H, W))."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % stride, (-w) % stride
    return torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect"), (h, w)
