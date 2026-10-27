"""Put a stack acquired with different detector settings on comp_full's grey scale.

The pore/solid rule (correct_phase_window.py) uses grey-level thresholds calibrated for comp_full.
comp_full and uncomp_full share a grey scale (cut faces saturate at 255, dark floor ~60-80), but
pristine_full was recorded with lower contrast (brightest cut faces ~190, median ~130). A linear map is
fitted on the two ends of the catalyst-layer histogram only, the darkest pore floor and the brightest
cut faces (percentiles `--low-pct` and `--high-pct` inside the layer), so that the mapping reflects the
detector, not the porosity (matching the whole histogram would impose comp_full's porosity).
`--gain-only` keeps zero fixed and matches only the bright end, for a sensitivity check.

Writes the mapped frames as one TIFF (frames 0..last+pad, same indices as the source) and relabels the
initial masks' pore pixels (< 140) on the mapped grey scale.

    python eponge/scripts/map_contrast.py --stack pristine_full.tif --masks ../runs/pristine/initial_masks \
        --out ../runs/pristine/mapped
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]


def layer_percentiles(stack_path, mask_dir, frames, pcts):
    vals = []
    st = tifffile.imread(stack_path, key=frames)
    for f, z in zip(st, frames):
        m = np.array(Image.open(Path(mask_dir) / f"slice_{z:04d}.png"))
        roi = ndi.binary_erosion(m != 128, iterations=10)
        vals.append(ndi.gaussian_filter(f.astype(np.float32), 1.0)[roi][::5])
    return np.percentile(np.concatenate(vals), pcts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--masks", required=True, help="initial masks (ROI) of this stack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-stack", default=str(ROOT / "data/raw/comp_full.tif"))
    ap.add_argument("--ref-masks", default=str(ROOT.parent / "runs/boundary_v7/corrected_masks"))
    ap.add_argument("--low-pct", type=float, default=2.0)
    ap.add_argument("--high-pct", type=float, default=98.0)
    ap.add_argument("--gain-only", action="store_true")
    ap.add_argument("--pad", type=int, default=7)
    ap.add_argument("--pore-below", type=float, default=140.0)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "initial_masks").mkdir(parents=True, exist_ok=True)
    masks = {int(re.fullmatch(r"slice_(\d+)\.png", p.name).group(1)): p for p in Path(a.masks).glob("slice_*.png")}
    frames = sorted(masks)
    ref_frames = [int(re.fullmatch(r"slice_(\d+)\.png", p.name).group(1)) for p in Path(a.ref_masks).glob("slice_*.png")]
    sample = frames[::10]
    ref_sample = sorted(ref_frames)[::10]
    src_lo, src_hi = layer_percentiles(a.stack, a.masks, sample, [a.low_pct, a.high_pct])
    ref_lo, ref_hi = layer_percentiles(a.ref_stack, a.ref_masks, ref_sample, [a.low_pct, a.high_pct])
    if a.gain_only:
        gain, offset = ref_hi / src_hi, 0.0
    else:
        gain = (ref_hi - ref_lo) / (src_hi - src_lo)
        offset = ref_lo - gain * src_lo
    n = max(frames) + 1 + a.pad
    with tifffile.TiffFile(a.stack) as t:
        n = min(n, len(t.pages))
    stack = tifffile.imread(a.stack, key=range(0, n)).astype(np.float32)
    mapped = np.clip(stack * gain + offset, 0, 255).round().astype(np.uint8)
    tifffile.imwrite(out / "mapped_stack.tif", mapped)
    for z in frames:
        m = np.array(Image.open(masks[z]))
        roi = m != 128
        sm = ndi.gaussian_filter(mapped[z].astype(np.float32), 1.0)
        new = np.full(m.shape, 128, np.uint8)
        new[roi] = np.where(sm[roi] < a.pore_below, 0, 255)
        Image.fromarray(new).save(out / "initial_masks" / f"slice_{z:04d}.png")
    info = {"gain": float(gain), "offset": float(offset), "gain_only": a.gain_only,
            "source_percentiles": [float(src_lo), float(src_hi)], "reference_percentiles": [float(ref_lo), float(ref_hi)],
            "percentiles_used": [a.low_pct, a.high_pct], "frames_written": n}
    (out / "mapping.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
