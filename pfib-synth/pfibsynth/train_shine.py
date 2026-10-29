#!/usr/bin/env python3
"""Train the team's U-Net (capstone_project_cmu_2026/model/unet.py) on
synthetic stacks, single-slice vs z-window, with an optional depth head.

Configs:
  2d_random        1 channel (the target slice), randomised physics
  zwin_random      9 channels (z-3 .. z+5, registered), randomised physics
  zwin_aux_random  as zwin_random, plus a head predicting how deep each
                   pixel's signal comes from (10 classes) -- a label only
                   synthetic data has
  zwin_narrow      9 channels, trained only on the calibrated comp physics

Same iteration budget, batch, crop size and data volume for every config.

    python train_shine.py --cache <dir> --config zwin_random --seed 0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
from paths import TEAM_REPO
# the team U-Net: a vendored copy here, else the team repo's own
UNET = next(p for p in (Path(__file__).resolve().parent / "team_unet.py", TEAM_REPO / "model" / "unet.py") if p.exists())
CONFIGS = {
    "2d_random":       {"data": "train_random", "window": False, "aux": False},
    "zwin_random":     {"data": "train_random", "window": True,  "aux": False},
    "zwin_aux_random": {"data": "train_random", "window": True,  "aux": True},
    "zwin_narrow":     {"data": "train_narrow_comp", "window": True, "aux": False},
    # v2 material (three-scale catalyst, SE grain and rim), same physics ranges
    "zwin_random_v2":  {"data": "train_random_v2", "window": True,  "aux": False},
    "2d_random_v2":    {"data": "train_random_v2", "window": False, "aux": False},
    # v3: see-through renderer (render_surface), random settings
    "zwin_random_v3":  {"data": "train_random_v3", "window": True,  "aux": False},
    "2d_random_v3":    {"data": "train_random_v3", "window": False, "aux": False},
    "zwin_random_v5":  {"data": "train_random_v5", "window": True,  "aux": False},
    "2d_random_v5":    {"data": "train_random_v5", "window": False, "aux": False},
    "zwin_random_v7":  {"data": "train_random_v7", "window": True,  "aux": False},
    "2d_random_v7":    {"data": "train_random_v7", "window": False, "aux": False},
}
CROP = 128
N_DEPTH = 10


def load_unet():
    spec = importlib.util.spec_from_file_location("team_unet", UNET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.UNet


def make_model(window, aux):
    UNet = load_unet()
    return UNet(in_channels=9 if window else 1, num_classes=1 + (N_DEPTH if aux else 0),
                base_ch=32)


def load_set(folder):
    Xs, Ps, Ds = [], [], []
    for f in sorted(Path(folder).glob("vol_*.npz")):
        d = np.load(f)
        Xs.append(d["X"]); Ps.append(d["pore"]); Ds.append(d["depth"])
    return np.concatenate(Xs), np.concatenate(Ps), np.concatenate(Ds)


def batches(X, P, D, window, rng, bs):
    n, _, h, w = X.shape
    while True:
        idx = rng.integers(0, n, bs)
        ys = rng.integers(0, h - CROP + 1, bs)
        xs = rng.integers(0, w - CROP + 1, bs)
        xb = np.stack([X[i, :, y:y + CROP, x:x + CROP] for i, y, x in zip(idx, ys, xs)]).astype(np.float32)
        pb = np.stack([P[i, y:y + CROP, x:x + CROP] for i, y, x in zip(idx, ys, xs)]).astype(np.float32)
        db = np.stack([D[i, y:y + CROP, x:x + CROP] for i, y, x in zip(idx, ys, xs)]).astype(np.int64)
        xb = np.clip(np.nan_to_num(xb, nan=0.0, posinf=20.0, neginf=-20.0), -20, 20)
        if not window:
            xb = xb[:, 3:4]
        # left-right flips only: an up-down flip would reverse the beam parallax
        flip = rng.random(bs) < 0.5
        xb[flip] = xb[flip][..., ::-1]; pb[flip] = pb[flip][..., ::-1]; db[flip] = db[flip][..., ::-1]
        # mild grey jitter on top of the per-stack normalisation
        xb = xb * rng.uniform(0.85, 1.15, (bs, 1, 1, 1)).astype(np.float32) \
             + rng.uniform(-0.15, 0.15, (bs, 1, 1, 1)).astype(np.float32)
        yield (torch.from_numpy(np.ascontiguousarray(xb)),
               torch.from_numpy(np.ascontiguousarray(pb)),
               torch.from_numpy(np.ascontiguousarray(db)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default=str(HERE / "checkpoints"))
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X, P, D = load_set(Path(args.cache) / cfg["data"])
    pos = float(P.mean())
    print(f"{args.config} seed {args.seed}: {X.shape[0]} frames, pore fraction {pos:.3f}, "
          f"device {dev}", flush=True)

    model = make_model(cfg["window"], cfg["aux"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.iters,
                                                pct_start=0.1)
    pw = torch.tensor([(1 - pos) / max(pos, 1e-3)], device=dev)
    gen = batches(X, P, D, cfg["window"], rng, args.bs)
    t0 = time.time(); run = 0.0
    model.train()
    for it in range(1, args.iters + 1):
        xb, pb, db = next(gen)
        xb, pb, db = xb.to(dev), pb.to(dev), db.to(dev)
        out = model(xb)
        loss = F.binary_cross_entropy_with_logits(out[:, 0], pb, pos_weight=pw)
        if cfg["aux"]:
            loss = loss + 0.3 * F.cross_entropy(out[:, 1:], db)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        lv = float(loss.detach())
        run = 0.98 * run + 0.02 * lv if it > 1 else lv
        if it % 250 == 0:
            print(f"  it {it:>5}  loss {run:.4f}  {time.time() - t0:.0f}s", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.config}_s{args.seed}.pt"
    torch.save({"state": model.state_dict(), "config": args.config, **cfg,
                "seed": args.seed, "iters": args.iters}, path)
    print(f"saved {path}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
