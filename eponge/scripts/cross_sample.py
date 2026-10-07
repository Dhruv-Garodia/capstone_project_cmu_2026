"""Cross-sample validation on the three pFIB-SEM stacks of Ferner et al. (2024).

Ferner et al. report porosity 41.7 % (PTL-fibre-compressed, comp_full), 51.5 % (pristine) and 55.8 %
(PTL-pore-area, uncomp_full). This script estimates porosity on all three stacks with every method,
on interior windows of the catalyst layer, to check ordering and magnitude on samples the twin models
never saw (they were trained on comp_full-calibrated twin data only).

    python eponge/scripts/cross_sample.py --uncomp PATH --pristine PATH
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

ROOT = Path(__file__).resolve().parents[2]
from eponge import ferner as FE  # noqa: E402
from eponge.inference import segment_stack  # noqa: E402
from eponge.metrics import largest_inner_rectangle  # noqa: E402
from eponge.models import load_checkpoint  # noqa: E402

PUBLISHED = {"comp": 0.417, "pristine": 0.515, "uncomp": 0.558}


def windows(roi_all, width=256, min_h=64, n=4):
    """Up to n non-overlapping width-px windows inside the layer for all slices, tallest first."""
    H, W = roi_all.shape
    pre = np.concatenate([np.zeros((H, 1), int), np.cumsum(roi_all, 1)], 1)
    cands = []
    for x0 in range(0, W - width + 1, width // 4):
        full = (pre[:, x0 + width] - pre[:, x0]) == width
        lab, k = ndi.label(full)
        if k:
            sizes = np.bincount(lab)[1:]
            i = int(np.argmax(sizes)) + 1
            ys = np.nonzero(lab == i)[0]
            cands.append((len(ys), x0, ys[0], ys[-1] + 1))
    cands.sort(reverse=True)
    out = []
    for h, x0, y0, y1 in cands:
        if h < min_h:
            break
        if all(abs(x0 - o[0]) >= width for o in out):
            out.append((x0, y0, y1))
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncomp", default="/Users/aadityavikram/Downloads/Experimental data/uncomp_full.tif")
    ap.add_argument("--pristine", default="/Users/aadityavikram/Downloads/Experimental data/pristine_full.tif")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    stacks = {"comp": ROOT / "data/raw/comp_full.tif", "uncomp": Path(a.uncomp), "pristine": Path(a.pristine)}
    roi_model, _ = load_checkpoint(ROOT / "runs/millnet_k0/best.pt", a.device)
    fbest = json.loads((ROOT / "runs/twin_real/ferner_oracle.json").read_text())["parameters"]
    twins = {f"MillNet 7-slice twin (real geom, seed {s})": ROOT / f"runs/twin_real/millnet_k3{'_s' + str(s) if s else ''}/best.pt" for s in (0, 1, 2)}
    twins["MillNet 7-slice twin (SliceGAN geom)"] = ROOT / "runs/twin_slicegan/millnet_k3/best.pt"
    twins.update({f"ResUNet 2D twin (real geom, seed {s})": ROOT / f"runs/twin_real/resunet_k0{'_s' + str(s) if s else ''}/best.pt" for s in (0, 1, 2)})
    twins["ResUNet 2D twin (SliceGAN geom)"] = ROOT / "runs/twin_slicegan/resunet_k0/best.pt"
    results = {}
    for name, path in stacks.items():
        with tifffile.TiffFile(path) as t:
            nz = len(t.pages)
        z0 = nz // 2 - a.n // 2
        st = tifffile.imread(path, key=range(z0, z0 + a.n))
        cls, _ = segment_stack(roi_model, st, 0, a.device, tta=False)
        roi_all = (cls != 2).all(0)
        wins = windows(roi_all)
        print(f"{name}: slices {z0}-{z0 + a.n - 1} of {nz}, frame {st.shape[1:]}, windows {[(x, y1 - y0) for x, y0, y1 in wins]}", flush=True)
        res = {"published_ferner_2024": PUBLISHED[name], "windows": [[int(v) for v in w] for w in wins]}
        boxes = [st[:, y0:y1, x0:x0 + 256] for x0, y0, y1 in wins]
        K = 3
        def por(fn):
            return float(np.mean([fn(b)[K:-K].mean() for b in boxes]))
        res["Otsu (denoised)"] = por(lambda b: np.stack([ndi.gaussian_filter(f.astype(np.float32), 1) <
                                                          threshold_otsu(ndi.gaussian_filter(f.astype(np.float32), 1)) for f in b]))
        res["Ferner, best-case parameters"] = por(lambda b: FE.segment(b, streak=1.0, **fbest))
        per_stack = {"MillNet 7-slice twin, calibrated to this acquisition": ROOT / f"runs/twin_{name}/millnet_k3_ft/best.pt"}
        for tn, ck in {**twins, **{k: v for k, v in per_stack.items() if v.exists()}}.items():
            m, c = load_checkpoint(ck, a.device)
            res[tn] = por(lambda b, m=m, c=c: segment_stack(m, b, c.get("k", 0), a.device, tta=True)[0] == 1)
        results[name] = res
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "windows"}, indent=1), flush=True)
    (ROOT / "runs/cross_sample.json").write_text(json.dumps(results, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
