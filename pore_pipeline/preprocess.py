"""Per-slice preprocessing: de-curtaining and histogram normalization.

destripe   -- wavelet-FFT stripe removal (Muench et al. 2009, Opt. Express 17:8567).
              Curtaining runs along the milling direction, vertical in these frames,
              so at each wavelet level the vertical-detail band is Fourier-filtered
              with a Gaussian notch along the stripe axis.
normalize  -- map each slice's ROI intensity histogram onto a reference slice's,
              which removes the brightness drift along z (Ferner et al. 2024 did the
              same). The LUT is fit on ROI pixels and applied to the whole frame.

RunFrames    -- the preprocessed frames of a finished run, recomputed on demand.
build_cache  -- the same frames for a whole stack as one uint8 .npy (memory-mapped),
                which is what the U-Net trains and predicts on.
"""
import json
from pathlib import Path

import numpy as np
import pywt
import tifffile

from . import labels as L

WAVELET = "db4"
LEVELS = None        # None = deepest level the frame allows (broad stripes need deep levels)
DAMP_SIGMA = 6.0     # px in frequency space; narrower = removes only near-perfect stripes


def destripe(img, wavelet=WAVELET, levels=LEVELS, sigma=DAMP_SIGMA):
    x = img.astype(np.float32)
    if levels is None:
        levels = pywt.dwt_max_level(min(x.shape), wavelet)
    coeffs = pywt.wavedec2(x, wavelet, level=levels)
    out = [coeffs[0]]
    for cH, cV, cD in coeffs[1:]:
        f = np.fft.fft2(cV)                     # vertical stripes live in the vertical-detail band
        f = np.fft.fftshift(f, axes=0)
        ky = np.arange(f.shape[0]) - f.shape[0] // 2
        damp = 1.0 - np.exp(-(ky ** 2) / (2 * sigma ** 2))
        f *= damp[:, None]
        cV = np.real(np.fft.ifft2(np.fft.ifftshift(f, axes=0))).astype(np.float32)
        out.append((cH, cV, cD))
    y = pywt.waverec2(out, wavelet)[: x.shape[0], : x.shape[1]]
    return np.clip(y, 0, 255)


def roi_cdf(img, roi):
    vals = img[roi]
    hist = np.bincount(np.clip(vals, 0, 255).astype(np.int64), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    return cdf / max(cdf[-1], 1)


def pick_reference(imgs, rois):
    """Reference = sampled slice whose ROI mean is the median across samples."""
    means = [img[roi].mean() if roi.any() else np.nan for img, roi in zip(imgs, rois)]
    return int(np.nanargmin(np.abs(np.array(means) - np.nanmedian(means))))


def normalize(img, roi, ref_cdf):
    """Histogram-match the frame so its ROI histogram matches ref_cdf."""
    if not roi.any():
        return img.astype(np.float32)
    cdf = roi_cdf(img, roi)
    lut = np.interp(cdf, ref_cdf, np.arange(256)).astype(np.float32)
    return lut[np.clip(img, 0, 255).astype(np.int64)]


class RunFrames:
    """De-striped, normalized frames of a stack, using a finished run's reference slice.

    The ROI used to fit each slice's histogram is the run's consensus mask, so the
    result matches what the pipeline segmented (up to the delamination gaps it removed).
    """

    def __init__(self, tif_path, run):
        self.tif = tifffile.TiffFile(tif_path)
        self.masks = Path(run) / "masks" / "consensus"
        ref_z = json.loads((Path(run) / "summary.json").read_text())["reference_slice"]
        self.ref_cdf = roi_cdf(destripe(self.tif.pages[ref_z].asarray()), self.roi(ref_z))

    def __len__(self):
        return len(self.tif.pages)

    def roi(self, z):
        return L.read_mask(self.masks, z) != L.EXCLUDED

    def __call__(self, z):
        return normalize(destripe(self.tif.pages[z].asarray()), self.roi(z), self.ref_cdf)


def build_cache(tif_path, run, name="preprocessed.npy"):
    """Write (or reuse) <run>/preprocessed.npy and return it memory-mapped, shape (Z, H, W) uint8."""
    path = Path(run) / name
    if path.exists():
        return np.load(path, mmap_mode="r")
    frames = RunFrames(tif_path, run)
    first = frames(0)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(frames), *first.shape))
    for z in range(len(frames)):
        out[z] = np.clip(np.rint(first if z == 0 else frames(z)), 0, 255)
    out.flush()
    return np.load(path, mmap_mode="r")
