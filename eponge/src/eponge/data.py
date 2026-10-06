"""Training data: random crops from labelled frames with physically sensible augmentation.

Augmentations (all in-plane; z is never flipped because milling direction is physical):
* horizontal flip (the 52 deg tilt geometry is symmetric left/right, not up/down);
* scale jitter 0.6-1.6x so the model tolerates other magnifications / pixel sizes;
* gamma, contrast, brightness, Gaussian blur, Poisson-like noise and vertical curtaining
  stripes so the model generalises to other microscopes and detector settings.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch.utils.data import Dataset

from .preprocess import normalize


def context_stack(stack: np.ndarray, z: int, k: int) -> np.ndarray:
    idx = np.clip(np.arange(z - k, z + k + 1), 0, len(stack) - 1)
    return stack[idx]


class CropDataset(Dataset):
    def __init__(self, stack: np.ndarray, labels: dict[int, np.ndarray], frames: list[int], k: int = 0,
                 crop: int = 256, n_samples: int = 4000, augment: bool = True, seed: int = 0):
        self.norm = np.stack([normalize(f) for f in stack]).astype(np.float32)  # per-frame normalisation
        self.labels = labels
        self.frames = [f for f in frames if f in labels]
        self.k, self.crop, self.n, self.augment = k, crop, n_samples, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = self.rng
        z = self.frames[rng.integers(len(self.frames))]
        img = context_stack(self.norm, z, self.k)
        lab = self.labels[z]
        scale = float(np.exp(rng.uniform(np.log(0.6), np.log(1.6)))) if self.augment else 1.0
        src = int(round(self.crop / scale))
        src = min(src, lab.shape[0], lab.shape[1])
        y = rng.integers(0, lab.shape[0] - src + 1)
        x = rng.integers(0, lab.shape[1] - src + 1)
        im = torch.from_numpy(np.ascontiguousarray(img[:, y:y + src, x:x + src]))[None]
        lb = torch.from_numpy(np.ascontiguousarray(lab[y:y + src, x:x + src]).astype(np.int64))[None, None].float()
        if src != self.crop:
            im = F.interpolate(im, size=(self.crop, self.crop), mode="bilinear", align_corners=False)
            lb = F.interpolate(lb, size=(self.crop, self.crop), mode="nearest")
        im, lb = im[0].numpy(), lb[0, 0].long().numpy()
        if self.augment:
            if rng.random() < 0.5:
                im, lb = im[:, :, ::-1], lb[:, ::-1]
            im = self._intensity(im, rng)
        return torch.from_numpy(np.ascontiguousarray(im)), torch.from_numpy(np.ascontiguousarray(lb))

    @staticmethod
    def _intensity(im: np.ndarray, rng) -> np.ndarray:
        im = im.copy()
        if rng.random() < 0.8:
            im = np.clip(im, 0, 1) ** float(np.exp(rng.uniform(-0.4, 0.4)))
        if rng.random() < 0.8:
            c, b = rng.uniform(0.75, 1.25), rng.uniform(-0.1, 0.1)
            im = (im - 0.5) * c + 0.5 + b
        if rng.random() < 0.3:
            s = rng.uniform(0.3, 1.2)
            im = np.stack([ndi.gaussian_filter(ch, s) for ch in im])
        if rng.random() < 0.5:
            im = im + rng.normal(0, rng.uniform(0.01, 0.08), im.shape)
        if rng.random() < 0.3:  # curtaining stripes
            stripes = rng.normal(0, rng.uniform(0.01, 0.05), im.shape[-1])
            im = im + ndi.gaussian_filter1d(stripes, rng.uniform(0.5, 3))[None, None, :]
        return np.clip(im, 0, 1).astype(np.float32)
