"""Label-free adaptation to an unseen acquisition, evaluated on the cross-stack (leave-one-stack-out) models.

A new stack arrives with images but no labels. Two standard adaptations use only those unlabelled images:

  histmatch  map each frame's grey levels onto the pooled histogram of the training stacks (test-time);
  adabn      re-estimate every BatchNorm layer's statistics on the new stack's unlabelled train-split frames
             (Li et al. 2016, "Revisiting batch normalization for practical domain adaptation").

No labels of the held-out stack are used; its test slices are only scored. Scoring is the same evaluate_frames
as everywhere else.

    python eponge/scripts/loso_adapt.py     # -> runs/final_models/loso_adapt.json
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from skimage.exposure import match_histograms

sys.path.insert(0, str(Path(__file__).parent))
import train_final as TF  # noqa: E402
from eponge.data import context_stack  # noqa: E402
from eponge.evaluate import evaluate_frames  # noqa: E402
from eponge.inference import _device, segment_stack  # noqa: E402
from eponge.models import load_checkpoint  # noqa: E402
from eponge.models.zoo import frame_quantiles  # noqa: E402
from eponge.preprocess import normalize  # noqa: E402

RUNS = TF.ROOT.parent / "runs/final_models"


def pooled_reference(data, stacks, n=24, seed=0):
    """A reference image whose histogram is the pooled training-stack histogram (sampled frames, concatenated)."""
    rng = np.random.default_rng(seed)
    parts = []
    for s in stacks:
        imgs, _, sp = data[s]
        for z in rng.choice(sp["train"], n // len(stacks), replace=False):
            parts.append(imgs[z].ravel())
    return np.concatenate(parts)[None]


def histmatch(images, ref):
    return np.stack([np.clip(np.round(match_histograms(f.astype(np.float32), ref.astype(np.float32))), 0, 255)
                     for f in images]).astype(np.uint8)


@torch.no_grad()
def adabn(model, images, frames, k, dev, n_batches=60, crop=512, bs=4, seed=0):
    """Recompute BatchNorm running statistics on unlabelled frames of the new stack."""
    m = copy.deepcopy(model)
    bns = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
    for b in bns:
        b.reset_running_stats(); b.momentum = None            # cumulative average over all batches
    m.eval()
    for b in bns:
        b.train()
    norm = np.stack([normalize(f) for f in images]).astype(np.float32)
    rng = np.random.default_rng(seed)
    H, W = norm.shape[1:]
    c = min(crop, H - H % 32, W - W % 32)
    for _ in range(n_batches):
        xs, qs = [], []
        for _ in range(bs):
            z = int(rng.choice(frames))
            y0, x0 = rng.integers(0, H - c + 1), rng.integers(0, W - c + 1)
            xs.append(context_stack(norm, z, k)[:, y0:y0 + c, x0:x0 + c])
            qs.append(frame_quantiles(torch.from_numpy(norm[z][None]))[0].numpy())
        x = torch.from_numpy(np.stack(xs)).to(dev)
        if getattr(m, "needs_stats", False):
            m(x, torch.from_numpy(np.stack(qs)).to(dev))
        else:
            m(x)
    m.eval()
    return m


def score(model, images, held, gts, k, dev):
    pr, _ = segment_stack(model, images, k, dev, tta=True, frames=held)
    m = evaluate_frames(dict(zip(held, pr)), gts)
    return {kk: m[kk] for kk in ("mean_iou", "pore_iou", "roi_iou", "porosity_mae")}


def main():
    dev = _device(None)
    data = {n: TF.load_stack(n) for n in TF.STACKS}
    out = {}
    for run in sorted(RUNS.glob("loso_*/best.pt")):
        name = run.parent.name
        held = name.split("_")[1]
        src = [s for s in TF.STACKS if s != held]
        model, ck = load_checkpoint(run, dev)
        k = ck.get("k", 3)
        imgs, masks, sp = data[held]
        test = [z for z in sp["test"] if z in masks]
        gts = {z: masks[z] for z in test}
        ref = pooled_reference(data, src)
        hm = histmatch(imgs, ref)
        res = {"none": score(model, imgs, test, gts, k, dev),
               "histmatch": score(model, hm, test, gts, k, dev)}
        ab = adabn(model, imgs, sp["train"], k, dev)
        res["adabn"] = score(ab, imgs, test, gts, k, dev)
        ab2 = adabn(model, hm, sp["train"], k, dev)
        res["histmatch+adabn"] = score(ab2, hm, test, gts, k, dev)
        out[name] = res
        print(name, {m: round(v["mean_iou"], 4) for m, v in res.items()}, flush=True)
        (RUNS / "loso_adapt.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
