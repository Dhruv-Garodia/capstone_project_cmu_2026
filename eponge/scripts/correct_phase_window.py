"""Correct the pore/solid labels inside the boundary-corrected ROI with a 5-slice shine-through rule.

The napari phase labels are a grey threshold (140-150, changing between annotators), so every pore that
shows material behind it (shine-through) is labelled solid. This script re-decides only the pixels the
threshold cannot decide, using the slices around each frame:

  * the layer darkens towards its edges (solid ~20 % dimmer within 10 px), so each slice is first
    brightness-corrected by distance from the edge, using the 90th grey-level percentile as the solid
    reference;
  * the final decision is the rule's 3D-smoothed result, as in Ferner et al.'s step 6 (--combine ferner).
    An earlier version (--combine conservative) then forced annotated-dark pixels to pore and very bright
    pixels to solid; that extra step is not in the paper and lowered porosity by 0.02-0.05;
  * (conservative only) dark pixels the annotators marked pore stay pore (unless only edge shading made them dark);
  * very bright pixels (cut faces) stay solid;
  * the grey pixels in between follow the Ferner et al. (2024) rule, computed over the stack: a pixel is
    a cut face only if it brightens sharply when the next slice is milled (forward z-gradient along the
    1 px/slice shine-through slide), smoothed in 3D over about 5 slices (sigma 1.5). Parameters are the
    ones tuned against exact truth on the digital twin (runs/twin_real/ferner_oracle.json), not fitted
    to any published porosity;
  * the boundary-corrected ROI (outside = 128) is kept as is; outside pixels are replaced by the nearest
    layer pixel before the rule, so the dark surroundings do not create a pore rim along the edges.

    python eponge/scripts/correct_phase_window.py
"""
import argparse
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

from eponge import ferner as FE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = [(0, 30), (31, 60), (61, 90), (91, 117)]
PUBLISHED = 0.417
EDGE_BINS = [0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40]


def edge_shading_gain(img: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Brightness gain that undoes the darkening of the layer towards its edges.

    The signal falls off near the layer surfaces: even the brightest (solid) pixels are ~20 % darker
    within 10 px of the edge. The 90th grey-level percentile, a proxy for solid brightness, is measured
    in bands of distance from the edge and scaled to the interior value (> 40 px)."""
    d = ndi.distance_transform_edt(roi)
    inner = roi & (d > EDGE_BINS[-1])
    inner = inner if inner.any() else roi
    ref90, ref50 = np.percentile(img[inner], [90, 50])
    centres, gains = [], []
    for lo, hi in zip(EDGE_BINS[:-1], EDGE_BINS[1:]):
        m = roi & (d > lo) & (d <= hi)
        if m.sum() > 200:
            p90, p50 = np.percentile(img[m], [90, 50])
            # never brighten the edge beyond its typical (median) darkening: where the edge only lacks
            # the brightest cut-face pixels (uncomp_full), the 90th percentile alone over-corrects
            g = min(ref90 / max(p90, 1.0), ref50 / max(p50, 1.0))
            centres.append((lo + hi) / 2)
            gains.append(np.clip(g, 1.0, 1.6))
    centres.append(EDGE_BINS[-1]); gains.append(1.0)
    return np.where(roi, np.interp(d, centres, gains), 1.0).astype(np.float32)


def side_shading_gain(img: np.ndarray, roi: np.ndarray, band: int = 64, step: int = 8) -> np.ndarray:
    """Brightness gain for the left/right ends of the layer, where a trench wall shades it.

    On pristine_full the layer ends inside the frame against trench walls and the solid within ~30 px of
    the wall is up to 20 % dimmer. Same rule as the edge gain: 90th percentile of 8-column bins scaled to
    the interior, capped by the median ratio, never below 1."""
    cols = np.nonzero(roi.any(0))[0]
    gain = np.ones(img.shape, np.float32)
    if len(cols) < 4 * band:
        return gain
    c0, c1 = cols.min(), cols.max()
    inner = roi.copy()
    inner[:, :c0 + band] = False
    inner[:, c1 - band + 1:] = False
    ref90, ref50 = np.percentile(img[inner], [90, 50])
    for start, sign in ((c0, 1), (c1, -1)):
        centres, gains = [], []
        for k in range(0, band, step):
            lo, hi = (start + k, start + k + step) if sign > 0 else (start - k - step + 1, start - k + 1)
            m = roi[:, lo:hi]
            if m.sum() > 100:
                p90, p50 = np.percentile(img[:, lo:hi][m], [90, 50])
                centres.append(k + step / 2)
                gains.append(np.clip(min(ref90 / max(p90, 1.0), ref50 / max(p50, 1.0)), 1.0, 1.6))
        centres.append(band); gains.append(1.0)
        dist = np.abs(np.arange(img.shape[1]) - start)
        g = np.interp(dist, centres, gains, right=1.0)
        side = (np.arange(img.shape[1]) >= c0) & (np.arange(img.shape[1]) <= c1)
        half = (np.arange(img.shape[1]) < (c0 + c1) / 2) if sign > 0 else (np.arange(img.shape[1]) >= (c0 + c1) / 2)
        gain[:, side & half] *= g[side & half]
    return np.where(roi, gain, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default=str(ROOT.parent / "runs/boundary_v7/corrected_masks"))
    ap.add_argument("--stack", default=str(ROOT / "data/raw/comp_full.tif"))
    ap.add_argument("--out", default=str(ROOT.parent / "runs/annotations_v3"))
    ap.add_argument("--streak", type=float, default=1.0)
    ap.add_argument("--fill", choices=["nearest", "mirror"], default="mirror",
                    help="how the outside is filled before the rule: mirror image of the layer across the "
                         "boundary (default) or the nearest layer pixel (earlier version; smears a featureless "
                         "band that the 3D smoothing turns into a pore rim where the layer ends inside the frame)")
    ap.add_argument("--side-gain", action="store_true",
                    help="also correct the shading of the layer's left/right ends by a trench wall (pristine)")
    ap.add_argument("--combine", choices=["ferner", "conservative"], default="ferner",
                    help="ferner: the rule's final 3D-smoothed decision, as in Ferner et al. step 6 (default); "
                         "conservative: also force dark pixels to pore and bright pixels to solid afterwards")
    ap.add_argument("--published", type=float, default=PUBLISHED, help="Ferner et al. 2024 porosity for this stack")
    ap.add_argument("--window", type=int, default=0,
                    help="slices the z-smoothing spans (5, 7, 11, ...): Gaussian sigma_z = window/4, in-plane "
                         "sigma unchanged; 0 keeps the tuned isotropic sigma 1.5 (about 5 slices)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    masks = {int(re.fullmatch(r"slice_(\d+)\.png", p.name).group(1)): np.array(Image.open(p))
             for p in Path(a.masks).glob("slice_*.png")}
    frames = sorted(masks)
    pad = max(6, a.window)                      # extra raw slices so every labelled frame has context
    stack = tifffile.imread(a.stack, key=range(0, max(frames) + 1 + pad))
    par = json.loads((ROOT / "runs/twin_real/ferner_oracle.json").read_text())["parameters"]
    nearest = [frames[int(np.argmin([abs(f - z) for f in frames]))] for z in range(len(stack))]
    sm = np.stack([ndi.gaussian_filter(f.astype(np.float32), par["sigma_2d"]) for f in stack])
    # edge shading correction (per slice, using the nearest labelled frame's ROI)
    gain = np.stack([edge_shading_gain(sm[z], masks[nearest[z]] != 128) for z in range(len(stack))])
    if a.side_gain:
        gain *= np.stack([side_shading_gain(sm[z], masks[nearest[z]] != 128) for z in range(len(stack))])
    flat = np.clip(stack.astype(np.float32) * gain, 0, 255)
    sm = sm * gain
    # The dark coating and PTL outside the layer would leak into the gradient, histogram matching and
    # 3D smoothing and mark a pore rim along every edge. Fill the outside of each slice with the nearest
    # layer pixel (unlabelled context slices use the nearest labelled frame's ROI) before the rule.
    filled = np.empty_like(flat)
    for z in range(len(stack)):
        out_roi = masks[nearest[z]] == 128
        idx = ndi.distance_transform_edt(out_roi, return_distances=False, return_indices=True)
        if a.fill == "mirror":
            grid = np.indices(out_roi.shape)
            mir = np.stack([np.clip(2 * idx[k] - grid[k], 0, out_roi.shape[k] - 1) for k in range(2)])
            ok = ~out_roi[tuple(mir)]                 # mirrored point must land inside the layer
            src = np.where(ok, mir, idx)
            filled[z] = flat[z][tuple(src)]
        else:
            filled[z] = flat[z][tuple(idx)]
    # Ferner et al.'s step 0 (match every slice's histogram to a reference slice), done on layer pixels
    # only: with whole frames the arbitrary content outside the layer moved interior porosity by ~0.05.
    rois = [masks[nearest[z]] != 128 for z in range(len(filled))]
    means = [filled[z][rois[z]].mean() for z in range(len(filled))]
    ref = int(np.argsort(means)[len(means) // 2])
    q = np.linspace(0, 1, 257)
    ref_q = np.quantile(filled[ref][rois[ref]], q)
    for z in range(len(filled)):
        filled[z] = np.interp(filled[z], np.quantile(filled[z][rois[z]], q), ref_q)
    par = {**par, "hist_match": False}
    if a.window:
        solid = ~FE.segment(filled, streak=a.streak, **{**par, "sigma_3d": 0.0})
        sig = (a.window / 4, par["sigma_3d"], par["sigma_3d"])
        window_pore = ~(ndi.gaussian_filter(solid.astype(np.float32), sig, mode="nearest") > 0.5)
    else:
        window_pore = FE.segment(filled, streak=a.streak, **par)

    rows = []
    for z in frames:
        lab, roi = masks[z], masks[z] != 128
        # annotated pores (I < 140-150) are certain, unless they were only dark because of edge shading
        dark = (lab == 0) & (sm[z] < 150)
        bright = sm[z] > par["high"]                # cut faces: certain solid
        pore = window_pore[z] if a.combine == "ferner" else dark | (~bright & window_pore[z])
        new = np.full_like(lab, 128)
        new[roi] = np.where(pore[roi], 0, 255)
        Image.fromarray(new).save(out / "masks" / f"slice_{z:04d}.png")
        rows.append({"frame": z, "roi_px": int(roi.sum()),
                     "annotated": float((lab == 0)[roi].mean()),
                     "annotated_pores_dropped_edge_shading": float(((lab == 0) & ~dark)[roi].mean()),
                     "window_rule_only": float(window_pore[z][roi].mean()),
                     "corrected": float(pore[roi].mean()),
                     "flipped_solid_to_pore": float((pore & (lab != 0))[roi].mean())})

    w = np.array([r["roi_px"] for r in rows], float)
    pooled = lambda k: float((np.array([r[k] for r in rows]) * w).sum() / w.sum())
    summary = {"frames": len(rows), "parameters": par, "streak": a.streak, "window": a.window, "combine": a.combine, "fill": a.fill,
               "porosity": {k: pooled(k) for k in ("annotated", "window_rule_only", "corrected")},
               "flipped_solid_to_pore": pooled("flipped_solid_to_pore"),
               "annotated_pores_dropped_edge_shading": pooled("annotated_pores_dropped_edge_shading"),
               "published_ferner_2024": a.published, "sessions": {}}
    for s0, s1 in SESSIONS:
        rs = [r for r in rows if s0 <= r["frame"] <= s1]
        summary["sessions"][f"{s0}-{s1}"] = {k: float(np.mean([r[k] for r in rs])) for k in ("annotated", "corrected")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "per_frame.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "parameters"}, indent=1))

    # figure 1: porosity per frame
    zf = [r["frame"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(zf, [r["annotated"] for r in rows], ".-", ms=3, lw=1, label=f"annotated  {summary['porosity']['annotated']:.3f}")
    ax.plot(zf, [r["corrected"] for r in rows], ".-", ms=3, lw=1, label=f"corrected (5-slice rule)  {summary['porosity']['corrected']:.3f}")
    ax.axhline(a.published, color="tab:green", ls="--", lw=1, label=f"Ferner et al. 2024  {a.published}")
    for s0, _ in SESSIONS[1:]:
        ax.axvline(s0 - 0.5, color="0.6", ls=":", lw=0.8)
    ax.set(xlabel="frame (dotted: annotator handovers)", ylabel="porosity inside ROI", ylim=(0, 0.6))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "porosity_per_frame.png", dpi=130)

    # figure 2: one frame, raw / annotated / corrected, zoomed into the layer
    z = 60
    roi = masks[z] != 128
    ys = np.nonzero(roi.any(1))[0]
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, 600, 1200
    raw = stack[z][y0:y1, x0:x1]
    def overlay(lab):
        rgb = np.stack([raw] * 3, -1).astype(float)
        rgb[lab == 0] = [235, 50, 50]
        rgb[lab == 128] = rgb[lab == 128] * 0.35 + np.array([40, 90, 255]) * 0.65
        return rgb.astype(np.uint8)
    corr = np.array(Image.open(out / "masks" / f"slice_{z:04d}.png"))[y0:y1, x0:x1]
    fig, ax = plt.subplots(3, 1, figsize=(10, 8.4))
    for axi, im, t in zip(ax, [np.stack([raw] * 3, -1), overlay(masks[z][y0:y1, x0:x1]), overlay(corr)],
                          ["raw frame 60", "annotated (threshold 150)", "corrected with 5-slice rule"]):
        axi.imshow(im)
        axi.set_title(t, fontsize=10)
        axi.axis("off")
    fig.tight_layout()
    fig.savefig(out / "frame60_comparison.png", dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
