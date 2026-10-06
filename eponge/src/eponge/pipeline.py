"""One-call pipeline: image or stack -> segmentation -> metrics -> uncertainty -> figures.

    >>> import eponge
    >>> res = eponge.analyze_file("slice.png", pixel_nm=6.0)
    >>> res.metrics["porosity"], res.uncertainty["porosity_mc"]
    >>> res.save("out/")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import metrics as M
from . import viz
from .inference import mc_dropout_porosity, segment_image, segment_stack, threshold_band
from .io import load_image, load_stack, save_mask
from .models import load_checkpoint
from .preprocess import normalize, register_stack  # register_stack only on request (see preprocess docs)

WEIGHTS = Path(__file__).parent / "weights"


def default_model(kind: str = "2d", device: str | None = None):
    name = {"2d": "eponge_millnet2d.pt", "2.5d": "eponge_millnet25d_k3.pt"}[kind]   # see docs/MILLNET.md
    path = WEIGHTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - train a model (eponge train) or copy runs/*/best.pt here")
    return load_checkpoint(path, device or "cpu")


@dataclass
class Result:
    image: np.ndarray                 # (H, W) or (Z, H, W) uint8
    classes: np.ndarray               # same shape, 0 solid / 1 pore / 2 outside
    pore_prob: np.ndarray             # same shape, float
    pixel_nm: float
    metrics: dict = field(default_factory=dict)
    uncertainty: dict = field(default_factory=dict)

    def save(self, out_dir: str | Path, make_3d: bool = True) -> Path:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        if self.classes.ndim == 2:
            save_mask(out / "mask.png", self.classes)
            img2d, cls2d = self.image, self.classes
        else:
            for z, c in enumerate(self.classes):
                save_mask(out / "masks" / f"slice_{z:04d}.png", c)
            mid = len(self.classes) // 2
            img2d, cls2d = self.image[mid], self.classes[mid]
        viz.plot_segmentation_map(img2d, cls2d, self.pixel_nm, out / "viz1_segmentation_map.png")
        viz.plot_quantitative(self.metrics, out / "viz2_quantitative.png", self.uncertainty.get("profile_band"))
        if self.classes.ndim == 3 and make_3d:
            viz.render_3d(self.classes, out / "viz3_3d_pore_network.html", self.pixel_nm)
        else:
            viz.plot_connectivity(cls2d, out / "viz3_connectivity.png")
        (out / "metrics.json").write_text(json.dumps({"metrics": self.metrics, "uncertainty": self.uncertainty},
                                                     indent=2, default=float))
        return out


def analyze_image(image: np.ndarray, model=None, pixel_nm: float = 6.0, mc_samples: int = 12,
                  device: str | None = None, tortuosity: bool = True) -> Result:
    if model is None:
        model, _ = default_model("2d", device)
    classes, probs = segment_image(model, image, device)
    res = Result(image, classes, probs[1], pixel_nm)
    res.metrics = M.analyze(classes, pixel_nm, compute_tortuosity=tortuosity)
    if mc_samples:
        res.uncertainty["porosity_mc"] = mc_dropout_porosity(model, normalize(image)[None], mc_samples, device)
    res.uncertainty["porosity_vs_threshold"] = threshold_band(probs)
    return res


def analyze_stack(stack: np.ndarray, model=None, k: int | None = None, pixel_nm: float = 6.0,
                  register: bool = False, device: str | None = None, tortuosity: bool = True) -> Result:
    if model is None:  # stacks: the 2.5D model (7 slices of context) unless the stack is too short
        model, ck = default_model("2.5d" if len(stack) >= 7 and k != 0 else "2d", device)
        k = ck.get("k", 0)
    k = k or 0
    if register:
        stack, shifts = register_stack(stack)
    classes, ppore = segment_stack(model, stack, k, device)
    res = Result(stack, classes, ppore.astype(np.float32), pixel_nm)
    res.metrics = M.analyze(classes, pixel_nm, compute_tortuosity=tortuosity)
    per_slice = [float((c[c != 2] == 1).mean()) for c in classes if (c != 2).any()]
    res.uncertainty["porosity_per_slice"] = {"mean": float(np.mean(per_slice)), "std": float(np.std(per_slice)),
                                             "p10": float(np.percentile(per_slice, 10)),
                                             "p90": float(np.percentile(per_slice, 90))}
    return res


def analyze_file(path: str | Path, **kw) -> Result:
    path = Path(path)
    k = kw.pop("k", None)
    image_kw = {key: v for key, v in kw.items() if key not in ("register",)}
    if path.is_dir():
        return analyze_stack(load_stack(path), k=k, **kw)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile
        with tifffile.TiffFile(path) as t:
            if len(t.pages) > 1:
                return analyze_stack(load_stack(path), k=k, **kw)
    return analyze_image(load_image(path), **image_kw)
