"""Train and evaluate any zoo architecture (2D or 2.5D) on the labelled pFIB-SEM frames.

Example:
    eponge train --stack data/raw/comp_full.tif --labels data/annotations_v2 \
        --out runs/transunet_k0 --arch transunet --k 0

Every architecture gets the same budget (iterations x batch, crop size, augmentation, split);
only the learning rate differs (high for networks trained from scratch, low for pretrained
transformer encoders). The checkpoint is chosen on validation frames 79-90 only; the test
frames 100-117 are evaluated once at the end.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CropDataset
from .evaluate import evaluate_frames
from .inference import _device, segment_stack
from .io import export_to_classes, load_masks, load_stack, save_mask
from .models import ARCHS, build_model, count_parameters

DEFAULT_LR = {"millnet": 2e-3, "resunet": 2e-3, "unet_garodia": 1e-3, "unetpp_r34": 5e-4, "segformer": 2e-4, "transunet": 1e-4}


def parse_frames(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = map(int, part.split("-")); out += range(a, b + 1)
        elif part.strip():
            out.append(int(part))
    return out


def dice_ce_loss(logits, target, weights):
    ce = F.cross_entropy(logits, target, weight=weights)
    p = logits.softmax(1)
    oh = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3)); den = (p + oh).sum((0, 2, 3))
    return ce + (1 - ((2 * inter + 1) / (den + 1)).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train eponge U-Net")
    ap.add_argument("--stack", required=True)
    ap.add_argument("--labels", required=True, help="folder of 0/128/255 masks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="resunet", choices=ARCHS)
    ap.add_argument("--k", type=int, default=0, help="adjacent slices each side (0 = 2D)")
    ap.add_argument("--no-bank", action="store_true", help="MillNet ablation: plain neighbour channels, no streak bank")
    ap.add_argument("--no-stats", action="store_true", help="MillNet ablation: no global histogram conditioning")
    ap.add_argument("--no-axial", action="store_true", help="MillNet ablation: no axial attention")
    ap.add_argument("--scale-range", type=float, nargs=2, default=None,
                    help="random rescale range for crops (default 0.6 1.6; 0.8 1.25 for millnet)")
    ap.add_argument("--train", default="0-74")
    ap.add_argument("--val", default="79-90")
    ap.add_argument("--test", default="100-117")
    ap.add_argument("--n-frames", type=int, default=118)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None, help="default depends on --arch")
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--max-ch", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-labels", default=None, help="extra label folder to also evaluate the test split on")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = _device(args.device)
    t0 = time.time()
    stack = load_stack(args.stack, range(args.n_frames))
    masks = {z: export_to_classes(m) for z, m in load_masks(args.labels).items() if z < args.n_frames}
    extra = {z: export_to_classes(m) for z, m in load_masks(args.eval_labels).items()} if args.eval_labels else None
    lr = args.lr or DEFAULT_LR[args.arch]
    tr, va, te = parse_frames(args.train), parse_frames(args.val), parse_frames(args.test)

    use_stats = args.arch == "millnet" and not args.no_stats
    # MillNet's streak bank assumes ~1 px/slice; keep scale augmentation narrow so the slide rate stays near it
    scale_range = tuple(args.scale_range) if args.scale_range else ((0.8, 1.25) if args.arch == "millnet" else (0.6, 1.6))
    ds = CropDataset(stack, masks, tr, k=args.k, crop=args.crop, n_samples=args.iters * args.bs, seed=args.seed,
                     stats=use_stats, scale_range=scale_range)
    dl = DataLoader(ds, batch_size=args.bs, num_workers=0)
    cfg = {"arch": args.arch, "in_channels": 2 * args.k + 1, "num_classes": 3}
    if args.arch == "resunet":
        cfg.update(base=args.base, depth=args.depth, dropout=0.1, max_ch=args.max_ch)
    if args.arch == "millnet":
        cfg.update(base=args.base, depth=args.depth, max_ch=args.max_ch, use_bank=not args.no_bank,
                   use_stats=use_stats, use_axial=not args.no_axial)
    model = build_model(cfg).to(dev)
    counts = np.sum([np.bincount(masks[z].ravel(), minlength=3) for z in tr if z in masks], 0)
    w = 1 / np.sqrt(counts / counts.sum()); w = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=args.iters, pct_start=0.1)
    print(f"device={dev} arch={args.arch} params={count_parameters(model)/1e6:.2f}M lr={lr} "
          f"train_frames={len(tr)} k={args.k} class_weights={w.tolist()}", flush=True)

    history, best, best_it = [], -1, 0
    model.train()
    for it, batch in enumerate(dl, 1):
        x, y = batch[0].to(dev), batch[1].to(dev)
        logits = model(x, batch[2].to(dev)) if use_stats else model(x)
        loss = dice_ce_loss(logits, y, w)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % 50 == 0:
            print(f"it {it} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
        if it % args.eval_every == 0 or it == args.iters:
            model.eval()
            cls, _ = segment_stack(model, stack, args.k, dev, tta=False, frames=va)
            m = evaluate_frames(dict(zip(va, cls)), {z: masks[z] for z in va})
            model.train()
            history.append({"iter": it, "loss": float(loss.item()), "val_mean_iou": m["mean_iou"],
                            "val_pore_iou": m["pore_iou"], "val_roi_iou": m["roi_iou"],
                            "val_porosity_mae": m["porosity_mae"]})
            print("  VAL", json.dumps(history[-1]), flush=True)
            if m["mean_iou"] > best:
                best, best_it = m["mean_iou"], it
                torch.save({"model": model.state_dict(), "config": model.config, "k": args.k,
                            "iter": it, "val": m, "label_set": args.labels},
                           out / "best.pt")
        if it >= args.iters:
            break

    ck = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    te = [z for z in te if z in masks]
    cls, _ = segment_stack(model, stack, args.k, dev, tta=True, frames=te)
    preds = dict(zip(te, cls))
    res = {"test_on_training_labels": evaluate_frames(preds, {z: masks[z] for z in te})}
    if extra:
        res["test_on_eval_labels"] = evaluate_frames(preds, {z: extra[z] for z in te if z in extra})
    for z, p in preds.items():
        save_mask(out / "test_predictions" / f"slice_{z:04d}.png", p)
    summary = {"args": {**vars(args), "lr": lr}, "best_iter": best_it, "params": count_parameters(model),
               "train_seconds": time.time() - t0, "history": history, **res}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    brief = {k: res["test_on_training_labels"][k] for k in ("mean_iou", "pore_iou", "roi_iou", "pore_f1_tol1",
                                                           "porosity_mae", "pred_porosity_mean", "gt_porosity_mean")}
    print("TEST", json.dumps(brief), "zcons", res["test_on_training_labels"]["z_consistency_pore_dice"], flush=True)


if __name__ == "__main__":
    main()
