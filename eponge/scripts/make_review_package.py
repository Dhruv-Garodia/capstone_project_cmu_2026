"""Build a four-person manual-review package (like team_review/ for comp_full) for another stack.

    python eponge/scripts/make_review_package.py --name uncomp --stack uncomp_full.tif \
        --masks ../runs/uncomp/masks_v3/masks --published 0.558 --start-porosity 0.514 \
        --out ../team_review_uncomp

Copies review_tool.py and combine.py from team_review/, writes start_masks/, a balanced assignment
(12 shared frames every 10th frame from 5; the rest in blocks of 5 rotating A-D, leftover frames
split so everyone's count differs by at most one, A last), overview sheets, stack.json and a README.
"""
import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT.parent / "team_review"


def assign(frames, shared):
    """Blocks of 5 frames rotating A-D; leftover frames split so counts differ by at most one, A last."""
    own = [z for z in frames if z not in shared]
    order = "ABCD"
    full = (len(own) // 20) * 20                       # whole rounds of A,B,C,D blocks
    rows = {z: order[(i // 5) % 4] for i, z in enumerate(own[:full])}
    rest = own[full:]
    extra = {n: len(rest) // 4 for n in order}
    for n in "BCD"[: len(rest) % 4]:
        extra[n] += 1
    i = 0
    for n in order:
        for z in rest[i:i + extra[n]]:
            rows[z] = n
        i += extra[n]
    counts = {n: sum(1 for v in rows.values() if v == n) for n in order}
    nxt = {"A": "B", "B": "C", "C": "D", "D": "A"}
    return [(z, "ALL", "-") if z in shared else (z, rows[z], nxt[rows[z]]) for z in frames], counts


def overlay(raw, lab):
    rgb = np.stack([raw] * 3, -1).astype(float)
    rgb[lab == 0] = [235, 50, 50]
    o = lab == 128
    rgb[o] = rgb[o] * 0.35 + np.array([40, 90, 255]) * 0.65
    return rgb.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--published", type=float, required=True)
    ap.add_argument("--start-porosity", type=float, required=True)
    ap.add_argument("--notes", default="", help="extra README paragraph for this stack")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "start_masks").mkdir(parents=True, exist_ok=True)
    (out / "overview").mkdir(exist_ok=True)
    for f in ("review_tool.py", "combine.py"):
        shutil.copy(TEMPLATE / f, out / f)
    masks = {int(re.fullmatch(r"slice_(\d+)\.png", p.name).group(1)): p for p in Path(a.masks).glob("slice_*.png")}
    frames = sorted(masks)
    for z, p in masks.items():
        shutil.copy(p, out / "start_masks" / f"slice_{z:04d}.png")
    shared = [z for z in range(5, max(frames) + 1, 10) if z in masks][:12]
    rows, counts = assign(frames, shared)
    with open(out / "assignment.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "annotator", "cross_checker"])
        w.writerows(rows)
    (out / "stack.json").write_text(json.dumps({"name": a.name, "stack_file": Path(a.stack).name,
                                                "published_porosity": a.published,
                                                "start_porosity": a.start_porosity}, indent=1))
    who = {z: o for z, o, _ in rows}
    st = tifffile.imread(a.stack, key=frames)
    tiles = []
    for raw, z in zip(st, frames):
        m = np.array(Image.open(masks[z]))
        roi = m != 128
        ys = np.nonzero(roi.any(1))[0]
        y0, y1 = max(ys.min() - 15, 0), min(ys.max() + 15, raw.shape[0])
        tiles.append((z, np.stack([raw] * 3, -1)[y0:y1:3, ::3], overlay(raw, m)[y0:y1:3, ::3], (m[roi] == 0).mean()))
    for s in range(0, len(tiles), 10):
        part = tiles[s:s + 10]
        H = max(t[1].shape[0] for t in part)
        W = part[0][1].shape[1]
        sheet = np.zeros((H * len(part), 2 * W + 6, 3), np.uint8)
        for i, (z, r, c, _) in enumerate(part):
            sheet[i * H:i * H + r.shape[0], :W] = r
            sheet[i * H:i * H + c.shape[0], W + 6:] = c
        im = Image.fromarray(sheet)
        d = ImageDraw.Draw(im)
        for i, (z, _, _, p) in enumerate(part):
            d.rectangle([0, i * H, 120, i * H + 14], fill=(0, 0, 0))
            d.text((3, i * H + 2), f"frame {z} raw", fill=(255, 255, 0))
            d.rectangle([W + 6, i * H, W + 230, i * H + 14], fill=(0, 0, 0))
            d.text((W + 9, i * H + 2), f"start mask {p:.3f} [{who[z]}]", fill=(255, 255, 0))
        im.save(out / "overview" / f"frames_{part[0][0]:03d}-{part[-1][0]:03d}.jpg", quality=85)
    readme = (TEMPLATE / "README.md").read_text()
    stack_file = Path(a.stack).name
    readme = readme.replace("# comp_full annotation review", f"# {a.name}_full annotation review")
    readme = readme.replace("comp_full.tif", stack_file).replace("comp_full", f"{a.name}_full")
    readme = re.sub(r"\| `start_masks/` \|[^\n]*\n",
                    f"| `start_masks/` | the masks we start from, {len(frames)} frames ({frames[0]}–{frames[-1]}) |\n", readme)
    readme = re.sub(r"## Where the start masks come from\n.*?(?=## The labelling rule)",
                    "## Where the start masks come from\n\n" + a.notes.strip() + "\n\n"
                    f"Start-mask porosity is **{a.start_porosity:.3f}** (Ferner et al. 2024 published "
                    f"{a.published} for this sample).\n\n", readme, flags=re.S)
    shared_txt = ", ".join(str(z) for z in shared[:3]) + ", …, " + str(shared[-1])
    readme = re.sub(r"On the 12 \*\*shared\*\* frames \([^)]*\)", f"On the 12 **shared** frames ({shared_txt})", readme)
    readme = re.sub(r"Ferner et al\. \(2024\) report 0\.417\.", f"Ferner et al. (2024) report {a.published}.", readme)
    (out / "README.md").write_text(readme)
    print(f"{out}: {len(frames)} frames, own frames per person {counts}, shared {shared}")


if __name__ == "__main__":
    main()
