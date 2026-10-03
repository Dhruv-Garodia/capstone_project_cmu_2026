"""Classical pipeline on one pFIB-SEM TIFF stack.

    python -m pore_pipeline.run data/comp_full.tif --out outputs/comp

Stages: de-curtain -> catalyst-layer ROI -> histogram normalization -> per-stack
calibration -> Otsu and Ferner-style segmentations -> consensus trimap.
Writes masks/{otsu,ferner_style,consensus}/slice_XXXX.png, stats.csv, summary.json
and preview.png. Label values are defined in labels.py.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from . import labels as L
from . import preprocess as P
from . import roi as R
from . import segment as S

CHUNK, HALO, ROI_STEP, N_SAMPLES = 32, 4, 3, 12
METHODS = ("otsu", "ferner_style", "consensus")


def detect_roi(get, z0, z1, shape, sample_z, margin):
    """Per-slice ROI function from boundaries found on every ROI_STEP-th slice."""
    tex_thr = R.texture_threshold([get(z) for z in sample_z])
    key_z = np.unique(np.r_[np.arange(z0, z1, ROI_STEP), z1 - 1])
    kt, kb = zip(*(R.raw_boundaries(get(z), tex_thr) for z in key_z))
    kt, kb = np.array(kt), np.array(kb)
    zs = np.arange(z0, z1)
    interp = lambda k: np.stack([np.interp(zs, key_z, k[:, x]) for x in range(shape[1])], 1)
    tops, bots = R.smooth_boundaries(interp(kt).astype(np.float32), interp(kb).astype(np.float32))
    return (lambda z: R.roi_from_boundaries(tops[z - z0], bots[z - z0], shape, margin=margin)), tex_thr


def calibrate_stack(norm, sample_z, rois, z1):
    """Thresholds, streak direction and gradient threshold, all measured from the stack."""
    cal = S.calibrate([norm(z) for z in sample_z], [rois[int(z)] for z in sample_z])
    inner = [int(z) for z in sample_z if z + 5 < z1]
    shift, n_blocks = S.estimate_streak_shift(norm, inner, rois)
    noise = S.gradient_noise([(norm(z), norm(z + 1)) for z in inner], [rois[z] for z in inner], shift)
    return cal, shift, n_blocks, S.GRAD_K * noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif")
    ap.add_argument("--out", required=True)
    ap.add_argument("--z0", type=int, default=0)
    ap.add_argument("--z1", type=int, default=None)
    ap.add_argument("--margin", type=int, default=R.MARGIN, help="px shrunk from the layer's top/bottom edge")
    ap.add_argument("--side-margin", type=int, default=R.SIDE_MARGIN, help="px trimmed at each lateral end")
    ap.add_argument("--no-destripe", action="store_true")
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:5.0f}s] {msg}")
    tif = tifffile.TiffFile(args.tif)
    z0, z1 = args.z0, min(args.z1 or len(tif.pages), len(tif.pages))
    shape = tif.pages[0].shape
    raw = lambda z: tif.pages[z].asarray()
    get = raw if args.no_destripe else (lambda z: P.destripe(raw(z)))
    out = Path(args.out)
    for m in METHODS:
        (out / "masks" / m).mkdir(parents=True, exist_ok=True)

    sample_z = np.unique(np.linspace(z0, z1 - 1, N_SAMPLES).astype(int))
    R.SIDE_MARGIN = args.side_margin
    roi_of, tex_thr = detect_roi(get, z0, z1, shape, sample_z, args.margin)
    log(f"ROI done (texture thr {tex_thr:.1f})")

    rois = {int(z): roi_of(z) for z in sample_z}
    samples = {int(z): get(z) for z in sample_z}
    ref_z = int(sample_z[P.pick_reference([samples[int(z)] for z in sample_z], [rois[int(z)] for z in sample_z])])
    ref_cdf = P.roi_cdf(samples[ref_z], rois[ref_z])
    norm = (lambda z: get(z).astype(np.float32)) if args.no_normalize else \
           (lambda z: P.normalize(get(z), roi_of(z), ref_cdf))
    log(f"reference slice {ref_z} (ROI mean {samples[ref_z][rois[ref_z]].mean():.1f})")

    cal, shift, n_blocks, grad_thr = calibrate_stack(norm, sample_z, rois, z1)
    log(f"calibration {cal}, streak shift {shift:+.2f} px/slice ({n_blocks} blocks), "
        f"gradient threshold {grad_thr:.1f}")

    rows = []
    for c0 in range(z0, z1, CHUNK):
        c1 = min(c0 + CHUNK, z1)
        h0, h1 = max(z0, c0 - HALO), min(z1, c1 + HALO)
        chunk = np.stack([S.smooth(norm(z)) for z in range(h0, h1)])
        ferner_solid = S.ferner_style(chunk, cal, shift, grad_thr)
        for z in range(c0, c1):
            g, fs = chunk[z - h0], ferner_solid[z - h0]
            roi = R.drop_edge_voids(roi_of(z), g < cal["low"])
            otsu_solid = g >= cal["otsu"]
            cons = S.consensus(otsu_solid, fs, roi)
            L.write_mask(out / "masks" / "otsu", z, S.to_label(otsu_solid, roi))
            L.write_mask(out / "masks" / "ferner_style", z, S.to_label(fs, roi))
            L.write_mask(out / "masks" / "consensus", z, cons)
            nroi = max(int(roi.sum()), 1)
            rows.append({"slice": z, "roi_fraction": roi.mean(),
                         "porosity_otsu": (roi & ~otsu_solid).sum() / nroi,
                         "porosity_ferner_style": (roi & ~fs).sum() / nroi,
                         "confident_pore": (cons == L.PORE).sum() / nroi,
                         "confident_solid": (cons == L.SOLID).sum() / nroi,
                         "uncertain": (cons == L.UNCERTAIN).sum() / nroi,
                         "mean_intensity_roi_raw": float(raw(z)[roi].mean()) if roi.any() else 0.0,
                         "mean_intensity_roi_norm": float(g[roi].mean()) if roi.any() else 0.0})
        log(f"slices {c0}-{c1 - 1}")

    with open(out / "stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    stat = lambda k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
    summary = {"stack": args.tif, "slices": [z0, z1], "texture_threshold": tex_thr, "calibration": cal,
               "margin": args.margin, "side_margin": args.side_margin,
               "destripe": not args.no_destripe, "normalize": not args.no_normalize, "reference_slice": ref_z,
               "streak_shift_px_per_slice": shift, "streak_blocks": n_blocks, "gradient_threshold": grad_thr,
               **{k: stat(k) for k in rows[0] if k != "slice"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    picks = [rows[i]["slice"] for i in np.linspace(0, len(rows) - 1, 4).astype(int)]
    tiles = []
    for z in picks:
        frame = np.clip(get(z), 0, 255).astype(np.uint8)
        tiles.append(np.concatenate([np.stack([frame] * 3, -1),
                                     L.overlay(frame, L.read_mask(out / "masks" / "consensus", z))], 1))
    Image.fromarray(np.concatenate(tiles, 0)).save(out / "preview.png")


if __name__ == "__main__":
    main()
