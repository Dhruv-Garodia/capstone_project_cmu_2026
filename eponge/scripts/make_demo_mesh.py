"""Build the comp_full 3D pore-network mesh shown in the web app (Visualisation 3).

1. Segment frames FIRST..FIRST+N-1 with the deployed 2D model (cached to runs/demo_segmentation.npy).
   Slices are stacked as acquired: the cut face does not drift (see eponge.preprocess).
2. Take the interior of the catalyst layer (rows inside the layer in every slice) and the
   480 px window whose porosity is closest to the whole layer's, so the demo is representative.
3. Mesh the pore phase at full resolution (eponge.mesh) and compute 3D metrics.

Writes webapp/samples/pore_mesh.{bin,json} (web viewer) and pore_mesh.glb (Blender/ParaView/any glTF viewer).

    python eponge/scripts/make_demo_mesh.py [--first 100 --n 128 --device cpu]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from eponge import mesh as MS
from eponge import metrics as M
from eponge.inference import segment_stack
from eponge.io import load_stack
from eponge.models import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/unet2d_v2/best.pt"))
    ap.add_argument("--first", type=int, default=100)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "webapp/samples"))
    a = ap.parse_args()

    cache = ROOT / f"runs/demo_segmentation_{a.first}_{a.n}.npy"
    if cache.exists():
        cls = np.load(cache)
    else:
        model, _ = load_checkpoint(a.ckpt, a.device)
        sub = load_stack(ROOT / "data/raw/comp_full.tif", range(a.first, a.first + a.n))
        cls, _ = segment_stack(model, sub, 0, a.device, tta=True, progress=True)
        np.save(cache, cls)

    layer_por = float((cls[cls != 2] == 1).mean())
    roi = (cls != 2).all(0)
    rs, cs = M.largest_inner_rectangle(roi)
    width = min(cs.stop - cs.start, a.width)
    col_por = (cls[:, rs, cs] == 1).mean((0, 1))
    win = np.convolve(col_por, np.ones(width) / width, mode="valid")
    c0 = cs.start + int(np.argmin(np.abs(win - layer_por)))
    pore = cls[:, rs, c0:c0 + width] == 1
    print(f"volume {pore.shape} (z, y, x) porosity {pore.mean():.3f} (whole layer {layer_por:.3f})", flush=True)

    res = MS.pore_mesh(pore, spacing_nm=(6.0, 6.0, 6.0))
    tau = M.tortuosity_factor(pore[:, :, :min(240, width)], axis=1)
    psd = M.pore_size_distribution(pore, pixel_nm=6.0)
    out = Path(a.out)
    extra = {
        "source": f"comp_full.tif frames {a.first}-{a.first + a.n - 1}, rows {rs.start}-{rs.stop}, columns {c0}-{c0 + width}",
        "layer_porosity": layer_por, "tau_through_plane": tau["tau"], "d50_nm": psd["d50"],
        "note": (f"{a.n} consecutive slices of comp_full (frames {a.first}–{a.first + a.n - 1}, "
                 f"{a.n * 6 / 1000:.2f} µm of milling) segmented by the 2D model and stacked as acquired. "
                 f"Window {width}×{rs.stop - rs.start} px inside the layer, chosen to match the whole layer's porosity. "
                 f"Porosity {pore.mean():.3f}, median pore diameter {psd['d50']:.0f} nm, through-plane tortuosity "
                 f"τ = {tau['tau']:.2f}, {100 * res['percolating_fraction']:.0f} % of pore volume connects top and bottom."),
    }
    meta = MS.save_web(res, out / "pore_mesh", extra)
    MS.save_glb(res, out / "pore_mesh.glb")
    print(json.dumps({k: v for k, v in meta.items() if k != "note"}, indent=1))
    print("glb", (out / "pore_mesh.glb").stat().st_size, "bin", (out / "pore_mesh.bin").stat().st_size)


if __name__ == "__main__":
    main()
