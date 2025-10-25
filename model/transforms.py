# model/transforms.py
from typing import Tuple, Callable
import random
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

class ToTensor:
    """Cofnvert PIL image to torch.Tensor [C,H,W], float32 in [0,1]."""
    def __call__(self, img):
        return TF.to_tensor(img)

class ComposePair:
    """
    Compose transforms for (image, mask) pairs.
    - 'both' transforms receive and must return (img, mask).
    - 'image_only' transforms receive and must return img.
    """
    def __init__(self, both: Tuple[Callable, ...] = (), image_only: Tuple[Callable, ...] = ()):
        self.both = both
        self.image_only = image_only

    def __call__(self, img, mask):
        for t in self.both:
            img, mask = t(img, mask)
        for t in self.image_only:
            img = t(img)
        return img, mask

class HFlip:
    """Paired horizontal flip with probability p."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        return img, mask

class VFlip:
    """Paired vertical flip with probability p."""
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        return img, mask

class RandomGamma:
    """
    Mild gamma correction on image only (mask untouched).
    Use a narrow range for stable, 'perfect' inputs.
    """
    def __init__(self, gamma_range=(0.9, 1.1), p: float = 0.0):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        if self.p <= 0.0:
            return img
        if random.random() < self.p:
            # Convert to tensor [0,1], apply power, then back to PIL
            t = TF.to_tensor(img).clamp(0, 1)
            g = random.uniform(*self.gamma_range)
            t = t.pow(g)
            img = TF.to_pil_image(t)
        return img
