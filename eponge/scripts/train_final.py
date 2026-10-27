"""Train and evaluate on the final four-annotator dataset (final_dataset/{comp,uncomp,pristine}).

    python eponge/scripts/train_final.py --arch millnet --k 3 --learned-slide --out runs/final/millnet_sa_k3

* Training stacks (--train-stacks) contribute equally many crops; checkpoints are chosen on the val
  slices of the training stacks only; each --test-stacks stack is scored once on its test slices.
* Metrics per test stack: 3-class mean IoU, pore IoU and layer IoU (evaluate_frames), porosity error,
  and **human-edit accuracy**: accuracy on the pixels where the annotators changed the automatic start
  masks. The start-mask generator scores 0 there by construction, so this is the part of the labels a
  model can only get right by learning what the humans corrected, not by imitating the generator.
* For models with a learned slide field, the mean predicted slide rate (px/slice) inside the layer is
  reported per stack, to compare with the independently measured rates.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader

from eponge.data import CropDataset, context_stack
from eponge.evaluate import evaluate_frames
from eponge.inference import _device, segment_stack
from eponge.io import export_to_classes
from eponge.models import build_model, count_parameters
from eponge.preprocess import normalize
from eponge.train import DEFAULT_LR, dice_ce_loss

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "final_dataset"
START = {"comp": ROOT.parent / "team_review/start_masks", "uncomp": ROOT.parent / "runs/merge_uncomp/start_masks",
         "pristine": ROOT.parent / "runs/merge_pristine/start_masks"}
STACKS = ("comp", "uncomp", "pristine")


def load_stack(name):
    d = DATA / name
    images = tifffile.imread(d / "images.tif")
    masks = {int(p.stem.split("_")[1]): export_to_classes(np.array(Image.open(p))) for p in (d / "masks").glob("slice_*.png")}
    split = json.loads((d / "split.json").read_text())
    return images, masks, split


class HalfCropDataset(CropDataset):
    """CropDataset that keeps the normalised stack in float16 (three stacks fit in memory)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.norm = self.norm.astype(np.float16)


def human_edit_accuracy(name, preds, gts):
    acc_n = acc_d = 0
    for z, p in preds.items():
        s = export_to_classes(np.array(Image.open(START[name] / f"slice_{z:04d}.png")))
        e = s != gts[z]
        acc_n += int((p[e] == gts[z][e]).sum()); acc_d += int(e.sum())
    return {"human_edited_px": acc_d, "human_edit_accuracy": acc_n / acc_d if acc_d else None}


def slide_rate(model, images, frames, k, dev):
    if not getattr(model, "learned_slide", False):
        return None
    norm = np.stack([normalize(f) for f in images]).astype(np.float32)
    vals = []
    with torch.no_grad():
        for z in frames[::3]:
            x = torch.from_numpy(context_stack(norm, z, k))[None].to(dev)
            h, w = x.shape[-2:]
            x = x[..., :h - h % 32, :w - w % 32]
            vals.append(float(model.slide(x).median()))
    return float(np.median(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="millnet")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--learned-slide", action="store_true")
    ap.add_argument("--encoder", default="resnet34", help="ZAF-UNet timm encoder")
    ap.add_argument("--fusion", default="zattn", choices=["zattn", "mean", "centre"], help="ZAF-UNet slice fusion (ablation)")
    ap.add_argument("--no-pretrain", action="store_true", help="random-initialised encoder (ablation)")
    ap.add_argument("--slide-max", type=float, default=2.5, help="MillNet-SA: bound on the learned slide rate (px/slice)")
    ap.add_argument("--slide-reg", type=float, default=0.0,
                    help="MillNet-SA: weight of a penalty on (slide/slide_max)^2, so the field moves only when it helps")
    ap.add_argument("--trace-v2", action="store_true", help="MillTrace: slide-aligned full-resolution path")
    ap.add_argument("--no-bank", action="store_true")
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--no-axial", action="store_true")
    ap.add_argument("--plain-head", action="store_true", help="MillNet ablation: plain 3-class head")
    ap.add_argument("--no-milling", action="store_true", help="MillNet ablation: raw neighbour slices, no milling channels")
    ap.add_argument("--train-frames", type=int, default=None,
                    help="few-shot: use only N evenly spaced train slices per stack and keep the last checkpoint "
                         "(no validation-based selection, so no labels beyond the N slices are used)")
    ap.add_argument("--train-stacks", default="comp,uncomp,pristine")
    ap.add_argument("--test-stacks", default="comp,uncomp,pristine")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--init", default=None, help="checkpoint to initialise from (matching tensors only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="pipeline test: 2 val/test slices per stack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = _device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trs, tes = a.train_stacks.split(","), a.test_stacks.split(",")
    data = {n: load_stack(n) for n in sorted(set(trs + tes))}
    is_mill = a.arch == "millnet"
    use_stats = is_mill and not a.no_stats
    scale_range = (0.8, 1.25) if is_mill else (0.6, 1.6)
    if a.train_frames:
        for n in trs:
            tr = data[n][2]["train"]
            data[n][2]["train"] = [tr[i] for i in np.linspace(0, len(tr) - 1, a.train_frames).round().astype(int)]
            print(f"few-shot {n}: train slices {data[n][2]['train']}", flush=True)
        a.eval_every = a.iters
    per = a.iters * a.bs // len(trs) + a.bs
    ds = ConcatDataset([HalfCropDataset(data[n][0], data[n][1], data[n][2]["train"], k=a.k, crop=a.crop, n_samples=per,
                                        seed=a.seed + 17 * i, stats=use_stats, scale_range=scale_range)
                        for i, n in enumerate(trs)])
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(a.seed))
    cfg = {"arch": a.arch, "in_channels": 2 * a.k + 1, "num_classes": 3}
    if a.arch == "resunet":
        cfg.update(base=24, depth=5, dropout=0.1, max_ch=128)
    if a.arch == "zafunet":
        cfg.update(encoder=a.encoder, fusion=a.fusion)
    if a.arch == "milltrace":
        cfg.update(slide_max=a.slide_max)
        if a.trace_v2:
            cfg.update(aligned_full=True, chs=(24, 24, 32, 64, 128))
    if a.no_pretrain:
        cfg["pretrained"] = False
    if is_mill:
        cfg.update(base=24, depth=5, max_ch=128, use_bank=not a.no_bank, use_stats=use_stats,
                   use_axial=not a.no_axial, learned_slide=a.learned_slide, slide_max=a.slide_max)
        if a.plain_head:
            cfg["factor_head"] = False
        if a.no_milling:
            cfg["use_milling"] = False
    model = build_model(cfg).to(dev)
    if a.init:
        sd = torch.load(a.init, map_location="cpu", weights_only=False)["model"]
        own = model.state_dict()
        ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(ok, strict=False)
        print(f"init from {a.init}: {len(ok)}/{len(own)} tensors", flush=True)
    counts = np.sum([np.bincount(m.ravel(), minlength=3) for n in trs for z, m in data[n][1].items()
                     if z in data[n][2]["train"]], 0)
    w = 1 / np.sqrt(counts / counts.sum()); w = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    lr = a.lr or {"zafunet": 4e-4, "unet_r34": 4e-4}.get(a.arch, DEFAULT_LR.get(a.arch, 1e-3))
    if hasattr(model, "enc"):          # pretrained encoder: 4x lower learning rate than the new layers
        enc_ids = {id(q) for q in model.enc.parameters()}
        groups = [{"params": [q for q in model.enc.parameters() if q.requires_grad], "lr": lr / 4},
                  {"params": [q for q in model.parameters() if id(q) not in enc_ids and q.requires_grad], "lr": lr}]
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr / 4, lr], total_steps=a.iters, pct_start=0.1)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=a.iters, pct_start=0.1)
    print(f"device={dev} arch={a.arch} k={a.k} slide={a.learned_slide} params={count_parameters(model)/1e6:.3f}M "
          f"lr={lr} train={trs} test={tes}", flush=True)

    def evaluate(names, split, tta):
        res = {}
        for n in names:
            images, masks, sp = data[n]
            fr = [z for z in sp[split] if z in masks]
            if a.smoke:
                fr = fr[:2]
            cls, _ = segment_stack(model, images, a.k, dev, tta=tta, frames=fr)
            preds, gts = dict(zip(fr, cls)), {z: masks[z] for z in fr}
            m = evaluate_frames(preds, gts)
            res[n] = {k: m[k] for k in ("mean_iou", "pore_iou", "roi_iou", "pore_f1_tol1", "porosity_mae",
                                        "pred_porosity_mean", "gt_porosity_mean")}
            if split == "test":
                res[n].update(human_edit_accuracy(n, preds, gts))
                res[n]["slide_rate_px_per_slice"] = slide_rate(model, images, fr, a.k, dev)
        return res

    best, hist = -1.0, []
    model.train()
    for it, batch in enumerate(dl, 1):
        x, y = batch[0].to(dev), batch[1].to(dev)
        logits = model(x, batch[2].to(dev)) if use_stats else model(x)
        loss = dice_ce_loss(logits, y, w)
        if a.slide_reg and getattr(model, "learned_slide", False):
            loss = loss + a.slide_reg * (model.last_slide / model.slide.slide_max).pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % 100 == 0:
            print(f"it {it} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if it % a.eval_every == 0 or it == a.iters:
            model.eval()
            v = evaluate(trs, "val", False)
            score = float(np.mean([r["mean_iou"] for r in v.values()]))
            hist.append({"iter": it, "val_mean_iou": score, **{n: r["mean_iou"] for n, r in v.items()}})
            print("  VAL", json.dumps(hist[-1]), flush=True)
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "config": model.config, "k": a.k, "iter": it}, out / "best.pt")
            model.train()
        if it >= a.iters:
            break
    ck = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    test = evaluate(tes, "test", True)
    summary = {"args": vars(a), "params": count_parameters(model), "best_iter": ck["iter"], "best_val": best,
               "train_seconds": time.time() - t0, "history": hist, "test": test}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for n, r in test.items():
        print(f"TEST {n}: mIoU {r['mean_iou']:.4f} pore {r['pore_iou']:.4f} layer {r['roi_iou']:.4f} "
              f"poro_err {r['porosity_mae']:.4f} human-edit acc {r['human_edit_accuracy']} slide {r['slide_rate_px_per_slice']}",
              flush=True)


if __name__ == "__main__":
    main()
