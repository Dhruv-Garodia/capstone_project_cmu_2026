"""Score every trained model and the non-learned baselines on the final dataset's test slices, one way.

    python eponge/scripts/eval_final.py            # writes runs/final_models/eval_all.json and prints a table

Metrics per test stack: 3-class mean IoU, pore IoU (inside the true layer), layer IoU, porosity error,
and accuracy on the pixels the annotators changed from the automatic start masks, split into
  * boundary edits: a pixel moved between "layer" and "outside";
  * phase edits: a pixel inside the layer changed between pore and solid (the shine-through decisions).
The start masks score 0 on both by construction; a baseline given the true boundary (Otsu) gets boundary
edits for free, so phase-edit accuracy is the fair comparison. Learned models also report their slide rate.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).parent))
import train_final as TF  # noqa: E402
from eponge.evaluate import evaluate_frames  # noqa: E402
from eponge.inference import _device, segment_stack  # noqa: E402
from eponge.io import export_to_classes  # noqa: E402
from eponge.models import load_checkpoint  # noqa: E402

OUT = TF.ROOT.parent / "runs/final_models"


def edit_accuracy(name, preds, gts):
    acc = {"boundary": [0, 0], "phase": [0, 0]}
    for z, p in preds.items():
        s = export_to_classes(np.array(Image.open(TF.START[name] / f"slice_{z:04d}.png")))
        g = gts[z]
        changed = s != g
        phase = changed & (s != 2) & (g != 2)
        boundary = changed & ~phase
        for k, m in (("boundary", boundary), ("phase", phase)):
            acc[k][0] += int((p[m] == g[m]).sum()); acc[k][1] += int(m.sum())
    return {f"{k}_edit_px": v[1] for k, v in acc.items()} | {f"{k}_edit_acc": (v[0] / v[1] if v[1] else None) for k, v in acc.items()}


def score(name, preds, gts):
    m = evaluate_frames(preds, gts)
    r = {k: m[k] for k in ("mean_iou", "pore_iou", "roi_iou", "porosity_mae", "pred_porosity_mean", "gt_porosity_mean")}
    return r | edit_accuracy(name, preds, gts)


def main():
    dev = _device(None)
    data = {n: TF.load_stack(n) for n in TF.STACKS}
    tests = {n: [z for z in data[n][2]["test"] if z in data[n][1]] for n in TF.STACKS}
    results = {"baselines": {}, "models": {}}
    for n in TF.STACKS:
        images, masks, _ = data[n]
        gts = {z: masks[z] for z in tests[n]}
        otsu, start = {}, {}
        for z in tests[n]:
            g = gts[z]; roi = g != 2
            sm = ndi.gaussian_filter(images[z].astype(float), 1)
            o = np.full(g.shape, 2, np.uint8); o[roi] = np.where(sm[roi] < threshold_otsu(sm[roi]), 1, 0); otsu[z] = o
            start[z] = export_to_classes(np.array(Image.open(TF.START[n] / f"slice_{z:04d}.png")))
        results["baselines"].setdefault("otsu_given_true_boundary", {})[n] = score(n, otsu, gts)
        results["baselines"].setdefault("start_masks_ferner_rule", {})[n] = score(n, start, gts)
    for run in sorted(OUT.iterdir()):
        ck, sm = run / "best.pt", run / "summary.json"
        if not (ck.exists() and sm.exists()) or run.name.startswith("_"):
            continue
        args = json.loads(sm.read_text())["args"]
        model, c = load_checkpoint(ck, dev)
        k = c.get("k", 0)
        res = {}
        for n in args["test_stacks"].split(","):
            images, masks, _ = data[n]
            cls, _ = segment_stack(model, images, k, dev, tta=True, frames=tests[n])
            res[n] = score(n, dict(zip(tests[n], cls)), {z: masks[z] for z in tests[n]})
            res[n]["slide_rate_px_per_slice"] = TF.slide_rate(model, images, tests[n], k, dev)
        results["models"][run.name] = {"train_stacks": args["train_stacks"], "test": res}
        print(run.name, {n: round(r["mean_iou"], 4) for n, r in res.items()}, flush=True)
    (OUT / "eval_all.json").write_text(json.dumps(results, indent=1))
    print("wrote", OUT / "eval_all.json")


if __name__ == "__main__":
    main()
