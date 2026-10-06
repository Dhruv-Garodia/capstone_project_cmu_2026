"""Inference: full-image / tiled prediction, flip TTA and Monte-Carlo-dropout uncertainty."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import context_stack
from .preprocess import normalize


def _device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _forward_padded(model, x: torch.Tensor, stats=None) -> torch.Tensor:
    m = getattr(model, "multiple", 32)
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
    y = model(xp, stats) if stats is not None else model(xp)
    return y[..., :h, :w]


def frame_stats(channels: np.ndarray, n_quant: int = 16) -> np.ndarray:
    """Whole-frame quantiles of the centre slice, the global context MillNet is conditioned on."""
    c = channels[len(channels) // 2]
    return np.quantile(c[::4, ::4], np.linspace(0.02, 0.98, n_quant)).astype(np.float32)[None]


def predict_probs(model, channels: np.ndarray, device: str | None = None, tta: bool = True,
                  tile: int = 1024, overlap: int = 128) -> np.ndarray:
    """Softmax probabilities (C, H, W) for normalised input ``channels`` of shape (K, H, W)."""
    device = _device(device)
    model = model.to(device)
    tile = getattr(model, "infer_tile", tile)
    st = (torch.from_numpy(frame_stats(channels, getattr(model, "n_quant", 16))).to(device)
          if getattr(model, "needs_stats", False) else None)
    overlap = min(overlap, tile // 4)
    x = torch.from_numpy(channels.astype(np.float32))[None].to(device)
    _, _, H, W = x.shape
    nc = getattr(model, "config", {}).get("num_classes", 3)
    out = torch.zeros((nc, H, W), device=device)
    cnt = torch.zeros((1, H, W), device=device)
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    with torch.no_grad():
        for y in ys:
            for x0 in xs:
                patch = x[..., y:y + tile, x0:x0 + tile]
                p = F.softmax(_forward_padded(model, patch, st), 1)
                if tta:
                    p = p + F.softmax(_forward_padded(model, patch.flip(-1), st), 1).flip(-1)
                    p = p / 2
                ph, pw = p.shape[-2:]
                out[:, y:y + ph, x0:x0 + pw] += p[0]
                cnt[:, y:y + ph, x0:x0 + pw] += 1
    return (out / cnt).cpu().numpy()


def segment_image(model, image: np.ndarray, device: str | None = None, tta: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """2D model on one uint8 image -> (class map, probabilities)."""
    probs = predict_probs(model, normalize(image)[None], device, tta)
    return probs.argmax(0).astype(np.uint8), probs


def segment_stack(model, stack: np.ndarray, k: int = 0, device: str | None = None, tta: bool = True,
                  frames: list[int] | None = None, progress: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Segment frames of a (registered) stack with a 2D (k=0) or 2.5D (k>0) model.

    Returns class maps (Z, H, W) uint8 and pore probabilities (Z, H, W) float16.
    """
    norm = np.stack([normalize(f) for f in stack]).astype(np.float32)
    frames = list(range(len(stack))) if frames is None else frames
    classes = np.zeros((len(frames),) + stack.shape[1:], np.uint8)
    ppore = np.zeros((len(frames),) + stack.shape[1:], np.float16)
    for i, z in enumerate(frames):
        p = predict_probs(model, context_stack(norm, z, k), device, tta)
        classes[i] = p.argmax(0)
        ppore[i] = p[1]
        if progress and i % 10 == 0:
            print(f"  segmented frame {z}", flush=True)
    return classes, ppore


def mc_dropout_porosity(model, channels: np.ndarray, n: int = 16, device: str | None = None,
                        roi: np.ndarray | None = None) -> dict:
    """Porosity distribution from Monte-Carlo dropout (dropout layers active, BN frozen).

    Returns median and the 10th-90th percentile band, following Norris et al. (Sandia 2024).
    """
    device = _device(device)
    model = model.to(device).eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout2d):
            m.train()
    vals = []
    try:
        for _ in range(n):
            p = predict_probs(model, channels, device, tta=False)
            cls = p.argmax(0)
            r = (cls != 2) if roi is None else roi
            vals.append(float((cls[r] == 1).mean()) if r.any() else float("nan"))
    finally:
        model.eval()
    vals = np.array(vals)
    return {"samples": vals.tolist(), "median": float(np.nanmedian(vals)),
            "p10": float(np.nanpercentile(vals, 10)), "p90": float(np.nanpercentile(vals, 90))}


def threshold_band(probs: np.ndarray, levels=(0.3, 0.4, 0.5, 0.6, 0.7)) -> dict:
    """Porosity when the pore decision threshold on p(pore | ROI) is varied (calibration sensitivity)."""
    roi = probs.argmax(0) != 2
    pp = probs[1] / np.maximum(probs[0] + probs[1], 1e-6)
    return {str(l): float((pp[roi] > l).mean()) for l in levels}
