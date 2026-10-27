"""Porosity of comp_full inside the boundary-corrected ROI, under every thresholding rule and model.

Uses Y. Lu's v7 boundary-corrected masks (`auto_correct_real_masks.py`, real-mask-boundary-correction
branch) to define the catalyst-layer ROI, then asks how porosity depends on the pore/solid rule:

* the annotated phase labels kept by the correction (napari threshold sessions);
* the grey-level threshold each frame's labels correspond to (reveals the annotation sessions);
* fixed thresholds 100-200, per-frame Otsu (raw and denoised), and one global Otsu;
* Éponge v2 labels, MillNet 2D (trained on v2), and MillNet 7-slice trained on the digital twin.

The twin model was trained on material-only images, and MillNet's whole-frame statistics and axial
attention see everything in its input, so it is run on the layer band (rows that are inside the ROI in
every frame and column) rather than on whole frames; on whole frames it under-reads porosity by ~0.2.
Every method is therefore reported on the full ROI and on that band.

    python eponge/scripts/boundary_porosity.py --masks ../runs/boundary_v7/corrected_masks
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage.filters import threshold_otsu  # noqa: E402

from eponge.inference import segment_stack  # noqa: E402
from eponge.models import load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = [(0, 30), (31, 60), (61, 90), (91, 117)]


def load_masks(d):
    out = {}
    for p in Path(d).glob("slice_*.png"):
        m = re.fullmatch(r"slice_(\d+)\.png", p.name)
        if m:
            out[int(m.group(1))] = np.array(Image.open(p))
    return dict(sorted(out.items()))


def label_threshold(img, lab, roi):
    """Grey threshold T that best reproduces 'pore = I < T' for one frame's labels, and its agreement."""
    v, p = img[roi].astype(int), lab[roi] == 0
    hp = np.bincount(v[p], minlength=257)
    hs = np.bincount(v[~p], minlength=257)
    # agreement(T) = pores below T + solids at or above T
    agree = np.cumsum(np.r_[0, hp])[:-1] + (hs.sum() - np.cumsum(np.r_[0, hs])[:-1])
    t = int(np.argmax(agree))
    return t, agree[t] / roi.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default=str(ROOT.parent / "runs/boundary_v7/corrected_masks"))
    ap.add_argument("--stack", default=str(ROOT / "data/raw/comp_full.tif"))
    ap.add_argument("--out", default=str(ROOT.parent / "runs/boundary_porosity"))
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    masks = load_masks(a.masks)
    v2 = load_masks(ROOT / "data/annotations_v2")
    frames = list(masks)
    stack = tifffile.imread(a.stack, key=range(0, max(frames) + 4))
    rois = {z: masks[z] != 128 for z in frames}
    roi_vals = np.concatenate([stack[z][rois[z]] for z in frames])
    t_global = float(threshold_otsu(roi_vals))

    m2d, c2d = load_checkpoint(ROOT / "runs/millnet_k0/best.pt", a.device)
    cls2d, _ = segment_stack(m2d, stack, 0, a.device, tta=True, frames=frames)
    # layer band: rows that are inside the ROI in (nearly) every column of every frame
    cover = np.min([r.mean(1) for r in rois.values()], axis=0)
    rows_in = np.nonzero(cover > 0.98)[0]
    y0, y1 = int(rows_in.min()), int(rows_in.max()) + 1
    mtw, ctw = load_checkpoint(ROOT / "runs/twin_slicegan/millnet_k3/best.pt", a.device)
    clstw, _ = segment_stack(mtw, stack[:, y0:y1], ctw.get("k", 3), a.device, tta=True, frames=frames)
    clstw_whole, _ = segment_stack(mtw, stack, ctw.get("k", 3), a.device, tta=True, frames=frames)

    fixed = list(range(100, 205, 5))
    rows = []
    for i, z in enumerate(frames):
        img, roi, lab = stack[z], rois[z], masks[z]
        n = roi.sum()
        sm = ndi.gaussian_filter(img.astype(np.float32), 1)
        t_lab, agree = label_threshold(img, lab, roi)
        band = np.zeros_like(roi)
        band[y0:y1] = True
        band &= roi
        lab_cls = {"annotated": lab == 0,
                   "otsu_raw": img < threshold_otsu(img[roi]),
                   "otsu_denoised": sm < threshold_otsu(sm[roi]),
                   "otsu_global": img < t_global,
                   "v2_labels": (v2[z] == 0) if z in v2 else None,
                   "millnet2d_v2": cls2d[i] == 1,
                   "millnet7_twin_wholeframe": clstw_whole[i] == 1}
        r = {"frame": z, "roi_px": int(n), "band_px": int(band.sum()),
             "label_threshold": t_lab, "label_threshold_agreement": float(agree),
             "otsu_raw_T": float(threshold_otsu(img[roi]))}
        for k, pore in lab_cls.items():
            r[k] = float(pore[roi].mean()) if pore is not None else np.nan
            r[k + "_band"] = float(pore[band].mean()) if pore is not None else np.nan
        r["millnet7_twin_band"] = float((clstw[i][band[y0:y1]] == 1).mean())
        for t in fixed:
            r[f"T{t}"] = float((img[roi] < t).mean())
        rows.append(r)

    with open(out / "per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def pooled(key):  # pixel-weighted porosity over all frames
        wts = np.array([r["band_px" if key.endswith("_band") else "roi_px"] for r in rows], float)
        vals = np.array([r[key] for r in rows], float)
        ok = ~np.isnan(vals)
        return float((vals[ok] * wts[ok]).sum() / wts[ok].sum())

    base = ["annotated", "otsu_raw", "otsu_denoised", "otsu_global", "v2_labels", "millnet2d_v2", "millnet7_twin_wholeframe"]
    methods = base + [m + "_band" for m in base] + ["millnet7_twin_band"]
    summary = {"frames": len(rows), "global_otsu_T": t_global, "band_rows": [y0, y1],
               "porosity": {k: pooled(k) for k in methods},
               "fixed_threshold": {t: pooled(f"T{t}") for t in fixed},
               "sessions": {}}
    for s0, s1 in SESSIONS:
        rs = [r for r in rows if s0 <= r["frame"] <= s1]
        summary["sessions"][f"{s0}-{s1}"] = {
            "frames": len(rs),
            "label_threshold_median": float(np.median([r["label_threshold"] for r in rs])),
            "label_threshold_range": [int(min(r["label_threshold"] for r in rs)), int(max(r["label_threshold"] for r in rs))],
            "label_agreement_mean": float(np.mean([r["label_threshold_agreement"] for r in rs])),
            "annotated_porosity": float(np.mean([r["annotated"] for r in rs])),
            "otsu_denoised_porosity": float(np.mean([r["otsu_denoised"] for r in rs])),
            "millnet7_twin_band_porosity": float(np.mean([r["millnet7_twin_band"] for r in rs]))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    # figure: porosity per frame by method, and porosity vs fixed threshold
    z = [r["frame"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    names = {"annotated": "annotated labels", "otsu_denoised": "Otsu (denoised)", "v2_labels": "Éponge v2 labels",
             "millnet2d_v2": "MillNet 2D (v2)", "millnet7_twin_band": "MillNet 7-slice (twin), layer band"}
    for k, lbl in names.items():
        ax[0].plot(z, [r[k] for r in rows], ".-", ms=3, lw=1, label=lbl)
    for s0, _ in SESSIONS[1:]:
        ax[0].axvline(s0 - 0.5, color="0.6", ls="--", lw=0.8)
    ax[0].set(xlabel="frame", ylabel="porosity inside corrected ROI", title="Porosity per frame")
    ax[0].legend(fontsize=8)
    ax[1].plot(z, [r["label_threshold"] for r in rows], "k.", ms=4, label="threshold matching annotation")
    ax[1].plot(z, [r["otsu_raw_T"] for r in rows], "-", color="tab:orange", lw=1, label="per-frame Otsu threshold")
    for s0, _ in SESSIONS[1:]:
        ax[1].axvline(s0 - 0.5, color="0.6", ls="--", lw=0.8)
    ax[1].set(xlabel="frame", ylabel="grey level", title="Which threshold the labels correspond to")
    ax[1].legend(fontsize=8)
    ax[2].plot(fixed, [summary["fixed_threshold"][t] for t in fixed], "o-", ms=3)
    for k, c in [("annotated", "k"), ("otsu_denoised", "tab:orange"), ("millnet7_twin_band", "tab:red")]:
        ax[2].axhline(summary["porosity"][k], color=c, ls=":", lw=1, label=f"{names[k]} {summary['porosity'][k]:.3f}")
    ax[2].axhline(0.417, color="tab:green", ls="--", lw=1, label="Ferner et al. 2024: 0.417")
    ax[2].set(xlabel="fixed threshold T (pore = I < T)", ylabel="porosity", title="Porosity vs threshold")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "porosity_vs_threshold.png", dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
