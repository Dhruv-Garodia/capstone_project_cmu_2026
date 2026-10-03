"""Train the 2.5D U-Net on one stack.

    python -m pore_pipeline.unet.train data/comp_full.tif outputs/comp \
        --labels outputs/comp/manual_borders/high --train-z 0-90 --val-z 100-117 \
        --k 3 --out outputs/comp/unet_k3

--k is the number of neighbouring slices on each side (k=0 is a plain 2D U-Net).
Train and validation slices must be separated along z: neighbouring slices are
near-duplicates, so a random split would leak. Loss = cross-entropy + Dice over
labelled pixels only. Writes best.pt, log.json and config.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .. import labels as L
from .. import preprocess as P
from .data import IGNORE, PatchDataset, parse_slices, to_target
from .model import UNet
from .predict import pick_device, pore_probability


def loss_fn(logits, target):
    valid = target != IGNORE
    ce = F.cross_entropy(logits, target, ignore_index=IGNORE)
    prob = torch.softmax(logits, 1)[:, 1][valid]
    truth = (target[valid] == 1).float()
    dice = 1 - (2 * (prob * truth).sum() + 1) / (prob.sum() + truth.sum() + 1)
    return ce + dice


def validate(model, frames, label_dir, zs, k, device):
    inter = union = correct = total = pred_pore = true_pore = 0
    for z, prob in pore_probability(model, frames, zs, k, device):
        target = to_target(L.read_mask(label_dir, z))
        valid = target != IGNORE
        pred, truth = prob[valid] >= 0.5, target[valid] == 1
        inter += (pred & truth).sum()
        union += (pred | truth).sum()
        correct += (pred == truth).sum()
        total += valid.sum()
        pred_pore += pred.sum()
        true_pore += truth.sum()
    return {"pore_iou": inter / union, "accuracy": correct / total,
            "porosity_pred": pred_pore / total, "porosity_label": true_pore / total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif")
    ap.add_argument("run")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--train-z", required=True)
    ap.add_argument("--val-z", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    frames = np.asarray(P.build_cache(args.tif, args.run))            # whole stack in RAM
    label_files = L.slice_files(args.labels)
    train_z = [z for z in parse_slices(args.train_z) if z in label_files]
    val_z = [z for z in parse_slices(args.val_z) if z in label_files]
    gap = min(abs(a - b) for a in val_z for b in train_z)
    print(f"device {device} | train {len(train_z)} slices, val {len(val_z)} slices, z-gap {gap} | k={args.k}")

    data = PatchDataset(frames, args.labels, train_z, args.k, args.patch, args.iters * args.batch, seed=args.seed)
    loader = DataLoader(data, batch_size=args.batch, num_workers=0)
    model = UNet(2 * args.k + 1, args.base, args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.1)

    log, best, running, t0 = [], -1.0, 0.0, time.time()
    for step, (x, t) in enumerate(loader, 1):
        model.train()
        loss = loss_fn(model(x.to(device)), t.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()
        if step % args.val_every == 0 or step == args.iters:
            model.eval()
            m = {k: float(v) for k, v in validate(model, frames, args.labels, val_z, args.k, device).items()}
            m.update(step=step, train_loss=running / args.val_every, seconds=round(time.time() - t0))
            running = 0.0
            log.append(m)
            print(f"step {step:5d}  loss {m['train_loss']:.4f}  val IoU {m['pore_iou']:.4f}  acc {m['accuracy']:.4f}  "
                  f"porosity pred {m['porosity_pred']:.3f} / label {m['porosity_label']:.3f}  [{m['seconds']}s]")
            if m["pore_iou"] > best:
                best = m["pore_iou"]
                torch.save({"model": model.state_dict(), "k": args.k, "base": args.base, "depth": args.depth,
                            "step": step, "val": m}, out / "best.pt")
            (out / "log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
