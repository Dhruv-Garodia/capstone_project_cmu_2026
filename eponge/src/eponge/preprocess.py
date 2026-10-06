"""Deterministic pre-processing for pFIB-SEM stacks.

* ``register_stack``: rigid slice-to-slice alignment by chained sub-pixel phase correlation.
  **Off by default and not used for comp_full.** Consecutive catalyst-layer crops correlate best
  at a ~1 px/slice y-offset, but that is sub-surface material sliding behind the cut face
  (shine-through, 52 deg geometry), not stage drift: cut-face structures outside the layer
  (PTL fibres, membrane) are stationary (|dy| <= 0.1 px/slice at every slice gap) and the
  offset is incoherent across gaps (-0.30, -1.27, 0.00 px/slice at gaps 1, 2, 5). Registering
  on it would misalign the real cut face by up to tens of pixels. Use only for stacks with
  verified stage drift, and verify on cut-face features (``residual_alignment``).
* ``destripe``: Fourier-domain removal of vertical curtaining stripes (Münch-style, FFT only).
* ``denoise``: light Gaussian denoising used for label correction and classical baselines.
* ``normalize``: robust percentile normalisation used as network input (same in the web app).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation


def normalize(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Map an image to float32 [0, 1] via its (low, high) percentiles. Robust to detector gain/offset."""
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [low, high])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def denoise(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return ndi.gaussian_filter(image.astype(np.float32), sigma)


def destripe(image: np.ndarray, sigma_ky: float = 2.0, keep_kx: int = 3) -> np.ndarray:
    """Suppress vertical curtaining stripes.

    Vertical stripes are constant along y, so their energy sits on the ky == 0 line of the
    2D spectrum. We attenuate a thin Gaussian band around that line while keeping the
    lowest |kx| <= keep_kx frequencies (global brightness gradients).
    """
    img = image.astype(np.float32)
    h, w = img.shape
    spec = np.fft.fft2(img - img.mean())
    ky = np.fft.fftfreq(h) * h
    kx = np.fft.fftfreq(w) * w
    band = np.exp(-(ky[:, None] ** 2) / (2 * sigma_ky ** 2))
    keep = (np.abs(kx)[None, :] <= keep_kx).astype(np.float32)
    filt = 1.0 - band * (1.0 - keep)
    out = np.real(np.fft.ifft2(spec * filt)) + img.mean()
    return out.astype(np.float32)


def estimate_shifts(stack: np.ndarray, rows: slice | None = None, upsample: int = 10) -> np.ndarray:
    """Cumulative (dy, dx) shift that aligns every frame onto frame 0.

    ``rows`` restricts the correlation to a band (e.g. the catalyst layer) so that
    featureless regions (Pt cap, membrane) do not dominate.
    """
    rows = rows if rows is not None else slice(None)
    shifts = np.zeros((len(stack), 2), dtype=np.float32)
    for k in range(1, len(stack)):
        s, _, _ = phase_cross_correlation(
            stack[k - 1][rows].astype(np.float32),
            stack[k][rows].astype(np.float32),
            upsample_factor=upsample,
        )
        shifts[k] = shifts[k - 1] + s
    return shifts


def apply_shifts(stack: np.ndarray, shifts: np.ndarray, order: int = 1) -> np.ndarray:
    """Apply per-frame shifts. Use ``order=0`` for label volumes."""
    out = np.empty_like(stack)
    for k, (frame, s) in enumerate(zip(stack, shifts)):
        out[k] = ndi.shift(frame, s, order=order, mode="nearest", prefilter=False)
    return out


def register_stack(stack: np.ndarray, rows: slice | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the registered stack and the per-frame cumulative shifts."""
    shifts = estimate_shifts(stack, rows)
    return apply_shifts(stack, shifts, order=1), shifts


def residual_alignment(stack: np.ndarray, rows: slice | None = None, step: int = 1) -> dict:
    """Report remaining frame-to-frame shift after registration (should be ~0)."""
    pair = [
        phase_cross_correlation(stack[k][rows or slice(None)].astype(np.float32),
                                stack[k + 1][rows or slice(None)].astype(np.float32),
                                upsample_factor=10)[0]
        for k in range(0, len(stack) - 1, step)
    ]
    pair = np.abs(np.array(pair))
    return {"mean_abs_dy": float(pair[:, 0].mean()), "mean_abs_dx": float(pair[:, 1].mean()),
            "max_abs_dy": float(pair[:, 0].max()), "max_abs_dx": float(pair[:, 1].max())}
