"""Best case of the Ferner et al. (2024) method: tune its six parameters on digital-twin data with exact
truth (validation volumes), score once on the held-out twin test volumes, then apply to real comp_full.

    python eponge/scripts/ferner_oracle.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from eponge import ferner as FE  # noqa: E402
from eponge import simulate as SIM  # noqa: E402
from eponge.io import load_masks, load_stack  # noqa: E402

out = ROOT / "runs/twin_real"
p = SIM.Imaging(**json.loads((out / "calibration.json").read_text())["params"])
cls = np.load(ROOT / "runs/twin/comp_full_millnet_classes.npy")
val = SIM.make_dataset(0, seed=2, p=p, geometries=SIM.real_geometry(cls, range(180, 250), 2, depth=64, seed=2))
test = SIM.make_dataset(0, seed=3, p=p, geometries=SIM.real_geometry(cls, range(250, 354), 4, depth=64, seed=3))

best, val_iou, log = FE.tune([v["images"] for v in val], [v["truth"] for v in val], streak=1.0, n_random=60)
print("best parameters (tuned on twin validation, exact truth):", {k: round(v, 3) for k, v in best.items()}, "val IoU", round(val_iou, 4), flush=True)

K = 3
P, T, aniso = [], [], []
for v in test:
    pr = FE.segment(v["images"], streak=1.0, **best)[K:-K]; tr = v["truth"][K:len(v["images"]) - K]
    P.append(pr.ravel()); T.append(tr.ravel())
P, T = np.concatenate(P), np.concatenate(T)
res = {"parameters": best, "val_pore_iou": val_iou, "test_pore_iou": float((P & T).sum() / (P | T).sum()),
       "test_porosity_pred": float(P.mean()), "test_porosity_true": float(T.mean())}
print("TEST (held-out real IrO2 geometry):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "parameters"}, flush=True)

# apply the oracle-tuned parameters to the real comp_full stack (same interior box as the twin experiment)
gf = load_masks(ROOT / "outputs/comp/masks/consensus")
stack = load_stack(ROOT / "data/raw/comp_full.tif", range(0, 130))
from eponge.metrics import largest_inner_rectangle  # noqa: E402
roi = np.ones(stack.shape[1:], bool)
for z in range(130):
    roi &= gf[z] != 128
rs, cs = largest_inner_rectangle(roi)
real = FE.segment(stack[:, rs, cs], streak=1.0, **best)[3:-3]
res["real_comp_full_porosity"] = float(real.mean())
res["ferner_2024_published_porosity_compressed_region"] = 0.417
print("real comp_full porosity with oracle-tuned Ferner parameters:", round(float(real.mean()), 4), "(published 0.417)", flush=True)
(out / "ferner_oracle.json").write_text(json.dumps({**res, "search_log": log}, indent=1))
