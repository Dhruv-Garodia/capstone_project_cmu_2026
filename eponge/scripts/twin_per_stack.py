"""Acquisition-aware digital twin: calibrate the twin to one stack and fine-tune MillNet 7-slice on it.

Stacks acquired with different beam/detector settings show different shine-through (comp_full slides
~1.2 px/slice, uncomp ~0.65, pristine ~0) and different grey-level response. A twin calibrated to one
acquisition is the wrong teacher for another, so for each stack:

  1. measure the sub-surface slide rate (phase correlation of neighbouring slices inside the layer);
  2. fit the twin's imaging parameters to the stack's catalyst-layer grey-level percentiles, keeping
     material behind the plane at least 20 grey levels darker than the cut face (without this the fit
     picks non-physical settings and the fine-tuned model over-reads porosity);
  3. render training volumes (real-derived + SliceGAN + random-field geometry) with that imaging;
  4. fine-tune MillNet 7-slice from the comp_full twin weights.

    python eponge/scripts/twin_per_stack.py --name pristine --stack PATH [--iters 1500]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from skimage.registration import phase_cross_correlation

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from eponge import simulate as SIM  # noqa: E402
from eponge.inference import _device, segment_stack  # noqa: E402
from eponge.models import load_checkpoint  # noqa: E402
import twin_experiment as TE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = _device(a.device)
    out = ROOT / f"runs/twin_{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    TE.OUT = out

    with tifffile.TiffFile(a.stack) as t:
        nz = len(t.pages)
    z0 = nz // 2 - 30
    st = tifffile.imread(a.stack, key=range(z0, z0 + 60))
    roi_model, _ = load_checkpoint(ROOT / "runs/millnet_k0/best.pt", dev)
    cls, _ = segment_stack(roi_model, st, 0, dev, tta=False)
    roi = (cls != 2)
    # 1. slide rate inside the layer band
    rows = np.nonzero(roi.mean((0, 2)) > 0.6)[0]
    band = slice(int(rows.min()), int(rows.max()) + 1)
    shifts = [phase_cross_correlation(st[i][band].astype(np.float32), st[i + 2][band].astype(np.float32),
                                      upsample_factor=10)[0][0] / 2 for i in range(0, 56, 2)]
    streak = float(-np.median(shifts))
    # 2. imaging calibration on the layer's grey values
    vals = st[roi][::7]
    geo_cls = np.load(ROOT / "runs/twin/comp_full_millnet_classes.npy")
    geo = SIM.real_geometry(geo_cls, range(0, 180), 1, depth=56, width=160, seed=99)[0]
    base = SIM.Imaging(streak=max(streak, 0.0))
    grid = {"depth_decay": [6, 10, 14, 18, 24, 32, 48], "back": [70, 100, 130, 160, 185, 215],
            "cut": [150, 180, 212, 232], "floor": [20, 45, 70, 90]}
    p, calib = SIM.calibrate(vals, grid=grid, solid=geo, min_contrast=20)
    p.streak = max(streak, 0.0)
    calib["params"]["streak"] = p.streak
    calib["measured_slide_px_per_slice"] = streak
    (out / "calibration.json").write_text(json.dumps(calib, indent=1))
    print(f"{a.name}: slide {streak:+.2f} px/slice; twin percentiles {np.round(calib['sim_percentiles'], 0).tolist()} "
          f"vs real {np.round(calib['target_percentiles'], 0).tolist()} (error {calib['mean_abs_error']:.1f})", flush=True)
    # 3. training data with this imaging
    gan_z = np.load(ROOT / "runs/slicegan/volumes.npz")
    gan = [gan_z[k] for k in sorted(gan_z.files, key=lambda s: int(s.split("_")[1]))]
    train = (SIM.make_dataset(0, seed=21, p=p, geometries=SIM.real_geometry(geo_cls, range(0, 180), 8, depth=64, seed=21))
             + SIM.make_dataset(0, seed=22, p=p, geometries=gan[:8]) + SIM.make_dataset(3, seed=23, p=p))
    val = SIM.make_dataset(0, seed=24, p=p, geometries=SIM.real_geometry(geo_cls, range(180, 250), 2, depth=64, seed=24))
    # 4. fine-tune MillNet 7-slice from the comp_full twin weights
    t0 = time.time()
    model, ck = TE.train_on_twin("millnet_k3_ft", {"arch": "millnet"}, 3, train, val, dev, a.iters, lr=5e-4,
                                 init=ROOT / "runs/twin_slicegan/millnet_k3/best.pt")
    print(f"fine-tuned in {time.time() - t0:.0f}s -> {out / 'millnet_k3_ft/best.pt'}", flush=True)


if __name__ == "__main__":
    main()
