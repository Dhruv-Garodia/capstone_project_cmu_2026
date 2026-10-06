"""Produce corrected annotations (v2) from the napari v1 masks and write an audit figure.

Usage:
    python scripts/make_annotations_v2.py --stack ../data/raw/comp_full.tif \
        --masks ../data/manual_annotation --out ../data/annotations_v2
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eponge.annotation import correct_folder, roi_z_consistency
from eponge.io import load_masks, load_stack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=118)
    args = ap.parse_args()

    stack = load_stack(args.stack, range(args.frames))
    summary = correct_folder(stack, args.masks, args.out)
    masks_v1 = load_masks(args.masks)
    zc = roi_z_consistency(masks_v1)

    zs = sorted(summary["frames"])
    rec = summary["frames"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    ax[0].plot(zs, [rec[z]["v1_porosity"] for z in zs], "o-", ms=3, label="v1 napari labels")
    ax[0].plot(zs, [rec[z]["v2_porosity"] for z in zs], "o-", ms=3, label="v2 corrected")
    ax[0].axhline(0.417, ls="--", c="k", lw=1, label="Ferner et al. 2024 (0.417)")
    ax[0].axhline(0.383, ls=":", c="gray", lw=1, label="Otsu on full stack (0.383)")
    ax[0].set(xlabel="frame", ylabel="ROI porosity", title="Porosity per annotated frame")
    ax[0].legend(fontsize=8)
    ax[1].step(zs, [rec[z]["v1_implied_raw_threshold"] for z in zs], where="mid", label="v1 implied raw threshold")
    ax[1].plot(zs, [rec[z]["v2_otsu_threshold_denoised"] for z in zs], label="v2 Otsu (denoised)")
    ax[1].set(xlabel="frame", ylabel="grey value", title="Threshold that defines 'pore'")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(args.out) / "annotation_audit.png", dpi=150)

    agg = {
        "v1_mean_porosity": float(np.mean([rec[z]["v1_porosity"] for z in zs])),
        "v2_mean_porosity": float(np.mean([rec[z]["v2_porosity"] for z in zs])),
        "v1_threshold_agreement_min": float(min(rec[z]["v1_threshold_agreement"] for z in zs)),
        "v1_thresholds_used": sorted({rec[z]["v1_implied_raw_threshold"] for z in zs}),
        "roi_dice_with_previous_min": float(min(zc.values())),
        "roi_dice_with_previous_frames_below_0.975": [z for z, d in zc.items() if d < 0.975],
    }
    (Path(args.out) / "audit_summary.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
