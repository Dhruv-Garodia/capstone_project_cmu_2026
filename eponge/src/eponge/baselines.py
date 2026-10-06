"""Classical baselines, run on identical frames for the comparison table.

These need a ROI (they cannot find the catalyst layer themselves); pass the reference ROI to
get an *oracle-ROI* upper bound for thresholding approaches.
"""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_local, threshold_otsu

from .io import OUTSIDE, PORE, SOLID
from .preprocess import denoise


def _compose(pore: np.ndarray, roi: np.ndarray) -> np.ndarray:
    out = np.full(roi.shape, OUTSIDE, np.uint8)
    out[roi] = SOLID
    out[pore & roi] = PORE
    return out


def otsu_raw(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    return _compose(image < threshold_otsu(image[roi]), roi)


def otsu_denoised(image: np.ndarray, roi: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    s = denoise(image, sigma)
    return _compose(s < threshold_otsu(s[roi]), roi)


def local_threshold(image: np.ndarray, roi: np.ndarray, block: int = 51, sigma: float = 1.0) -> np.ndarray:
    s = denoise(image, sigma)
    return _compose(s < threshold_local(s, block, offset=0), roi)


def fixed_threshold(image: np.ndarray, roi: np.ndarray, t: int = 140) -> np.ndarray:
    """What the v1 annotations implicitly are (raw I < 140..150)."""
    return _compose(image < t, roi)


BASELINES = {
    "otsu_raw": otsu_raw,
    "otsu_denoised": otsu_denoised,
    "local_threshold": local_threshold,
    "fixed_threshold_140": fixed_threshold,
}
