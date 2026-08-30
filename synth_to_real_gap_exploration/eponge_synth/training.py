"""
training.py — 2.5D / 3D pore-solid segmentation on simulated pFIB-SEM stacks.

    python -m eponge_synth.training --data datasets/v2 --arch deeplabv3plus --mode 2.5d --epochs 40
    python -m eponge_synth.training --data datasets/v2 --arch unet3d --mode 3d --patch 32 128 128

Data layout (produced by make_dataset.py):  <data>/stacks/<name>/{images.npy, masks.npy, meta.json}
    images.npy  (N, Y, X) uint8      masks.npy (N, Y, X) bool (1 = solid)
Losses:  CE + Dice (baseline)  +  boundary loss (Kervadec et al. 2019, signed-distance weighted)
         +  inter-slice consistency (predictions of z and z+1 must agree where the labels agree).
Augmentation respects the imaging frame: shine-through is displaced along +/-y, so we flip x freely
but flip y only when the dataset was rendered with random `beam_sign` (meta['beam_sign_random']).
No in-plane rotations: the slicing/beam frame is canonical in real pFIB-SEM data.
NOTE: written for a GPU box; not executed in the authoring sandbox (no torch there). Run smoke_test.py first.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def _load_stacks(root):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "stacks", "*"))):
        if os.path.exists(os.path.join(d, "images.npy")):
            meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
            out.append(dict(name=os.path.basename(d), images=np.load(os.path.join(d, "images.npy"), mmap_mode="r"),
                            masks=np.load(os.path.join(d, "masks.npy"), mmap_mode="r"), meta=meta))
    if not out:
        raise FileNotFoundError(f"no stacks under {root}/stacks")
    return out


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Negative inside the foreground, positive outside (Kervadec boundary loss convention)."""
    m = mask.astype(bool)
    if m.all() or not m.any():
        return np.zeros(m.shape, np.float32)
    inside = ndi.distance_transform_edt(m)
    outside = ndi.distance_transform_edt(~m)
    return (outside - inside).astype(np.float32)


class SliceDataset(Dataset):
    """2.5D: channels = [z-c .. z+c]; target = mask[z]; also returns mask[z+1] for the consistency loss."""

    def __init__(self, root, split="train", context=1, crop=128, val_frac=0.15, style=None, seed=0):
        self.stacks = _load_stacks(root)
        n_val = max(1, int(len(self.stacks) * val_frac))
        self.stacks = self.stacks[:-n_val] if split == "train" else self.stacks[-n_val:]
        self.context, self.crop, self.train = context, crop, split == "train"
        self.rng = np.random.default_rng(seed)
        self.style = style  # optional callable(img uint8) -> uint8 (real-style transfer)
        self.index = [(i, z) for i, s in enumerate(self.stacks) for z in range(s["images"].shape[0] - 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, z = self.index[k]
        s = self.stacks[i]
        imgs, masks = s["images"], s["masks"]
        N, Y, X = imgs.shape
        zs = [min(max(z + d, 0), N - 1) for d in range(-self.context, self.context + 1)]
        x = np.stack([imgs[j] for j in zs]).astype(np.float32)          # (C, Y, X)
        y = masks[z].astype(np.int64)
        y1 = masks[min(z + 1, N - 1)].astype(np.int64)
        if self.style is not None and self.train and self.rng.uniform() < 0.5:
            x = np.stack([self.style(c.astype(np.uint8)).astype(np.float32) for c in x])
        if self.train:
            cy, cx = self.rng.integers(0, max(Y - self.crop, 0) + 1), self.rng.integers(0, max(X - self.crop, 0) + 1)
            sl = (slice(cy, cy + self.crop), slice(cx, cx + self.crop))
            x, y, y1 = x[:, sl[0], sl[1]], y[sl], y1[sl]
            if self.rng.uniform() < 0.5:                                # x-flip: always frame-safe
                x, y, y1 = x[:, :, ::-1], y[:, ::-1], y1[:, ::-1]
            if s["meta"].get("beam_sign_random", False) and self.rng.uniform() < 0.5:
                x, y, y1 = x[:, ::-1, :], y[::-1, :], y1[::-1, :]
            x = x * self.rng.uniform(0.85, 1.15) + self.rng.uniform(-15, 15)   # gain/offset jitter
        x = np.ascontiguousarray((x / 255.0 - 0.35) / 0.25)
        sdf = signed_distance(y)
        return (torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(y1)), torch.from_numpy(sdf))


class PatchDataset3D(Dataset):
    def __init__(self, root, split="train", patch=(32, 128, 128), val_frac=0.15, n_per_stack=64, seed=0):
        self.stacks = _load_stacks(root)
        n_val = max(1, int(len(self.stacks) * val_frac))
        self.stacks = self.stacks[:-n_val] if split == "train" else self.stacks[-n_val:]
        self.patch, self.n = patch, n_per_stack
        self.rng = np.random.default_rng(seed)
        self.train = split == "train"

    def __len__(self):
        return len(self.stacks) * self.n

    def __getitem__(self, k):
        s = self.stacks[k // self.n]
        imgs, masks = s["images"], s["masks"]
        D, H, W = self.patch
        N, Y, X = imgs.shape
        z0, y0, x0 = (self.rng.integers(0, max(N - D, 0) + 1), self.rng.integers(0, max(Y - H, 0) + 1),
                      self.rng.integers(0, max(X - W, 0) + 1))
        x = imgs[z0:z0 + D, y0:y0 + H, x0:x0 + W].astype(np.float32)
        y = masks[z0:z0 + D, y0:y0 + H, x0:x0 + W].astype(np.int64)
        if self.train and self.rng.uniform() < 0.5:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        x = np.ascontiguousarray((x / 255.0 - 0.35) / 0.25)[None]      # (1, D, H, W)
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(y))


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------
class _DoubleConv(nn.Module):
    def __init__(self, i, o, conv=nn.Conv2d, bn=nn.BatchNorm2d):
        super().__init__()
        self.b = nn.Sequential(conv(i, o, 3, padding=1, bias=False), bn(o), nn.ReLU(inplace=True),
                               conv(o, o, 3, padding=1, bias=False), bn(o), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.b(x)


class UNet(nn.Module):
    """Compact U-Net, 2-D or 3-D (dim=2/3)."""

    def __init__(self, in_ch=3, n_classes=2, base=32, dim=2, depth=4):
        super().__init__()
        conv, bn = (nn.Conv2d, nn.BatchNorm2d) if dim == 2 else (nn.Conv3d, nn.BatchNorm3d)
        self.pool = nn.MaxPool2d(2) if dim == 2 else nn.MaxPool3d((1, 2, 2))
        self.dim, self.depth = dim, depth
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_DoubleConv(in_ch if i == 0 else chs[i - 1], chs[i], conv, bn) for i in range(depth + 1)])
        self.dec = nn.ModuleList([_DoubleConv(chs[i] + chs[i - 1], chs[i - 1], conv, bn) for i in range(depth, 0, -1)])
        self.out = conv(chs[0], n_classes, 1)

    def forward(self, x):
        feats = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else self.pool(x))
            feats.append(x)
        x = feats[-1]
        for i, d in enumerate(self.dec):
            skip = feats[-2 - i]
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear" if self.dim == 2 else "trilinear", align_corners=False)
            x = d(torch.cat([skip, x], 1))
        return self.out(x)


def build_model(arch: str, in_ch: int, n_classes: int = 2):
    if arch == "deeplabv3plus":
        try:
            import segmentation_models_pytorch as smp
            return smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=in_ch, classes=n_classes)
        except ImportError:
            print("[warn] segmentation_models_pytorch not installed -> falling back to UNet2D")
            return UNet(in_ch, n_classes, dim=2)
    if arch == "unet2d":
        return UNet(in_ch, n_classes, dim=2)
    if arch == "unet3d":
        return UNet(1, n_classes, base=24, dim=3, depth=3)
    raise ValueError(arch)


# --------------------------------------------------------------------------------------
# losses & metrics
# --------------------------------------------------------------------------------------
def dice_loss(logits, target, eps=1.0):
    p = logits.softmax(1)
    t = F.one_hot(target, logits.shape[1]).movedim(-1, 1).float()
    dims = tuple(range(2, logits.ndim))
    inter = (p * t).sum(dims)
    return (1 - (2 * inter + eps) / (p.sum(dims) + t.sum(dims) + eps)).mean()


def boundary_loss(logits, sdf):
    """Kervadec et al.: mean over pixels of p_fg * signed distance (negative inside fg)."""
    return (logits.softmax(1)[:, 1] * sdf).mean()


def consistency_loss(logits_z, logits_z1, y_z, y_z1):
    same = (y_z == y_z1).float()
    return ((logits_z.softmax(1)[:, 1] - logits_z1.softmax(1)[:, 1]).abs() * same).sum() / same.sum().clamp(min=1)


def total_loss(logits, y, sdf=None, w_ce=0.5, w_dice=0.5, w_bd=0.0, w_cons=0.0, logits_next=None, y_next=None):
    loss = w_ce * F.cross_entropy(logits, y) + w_dice * dice_loss(logits, y)
    if w_bd > 0 and sdf is not None:
        loss = loss + w_bd * boundary_loss(logits, sdf)
    if w_cons > 0 and logits_next is not None:
        loss = loss + w_cons * consistency_loss(logits, logits_next, y, y_next)
    return loss


@torch.no_grad()
def metrics(logits, y):
    pred = logits.argmax(1)
    fg_p, fg_t = pred == 1, y == 1
    inter = (fg_p & fg_t).sum().item()
    union = (fg_p | fg_t).sum().item()
    iou = inter / max(union, 1)
    dice = 2 * inter / max(fg_p.sum().item() + fg_t.sum().item(), 1)
    pix = (pred == y).float().mean().item()
    por_err = abs((1 - fg_p.float().mean()) - (1 - fg_t.float().mean())).item()   # porosity error
    return dict(iou=iou, dice=dice, pixacc=pix, porosity_abs_err=por_err)


# --------------------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------------------
def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style = None
    if args.real_refs:
        from .domain_adapt import random_real_style
        from PIL import Image
        refs = [np.array(Image.open(p).convert("L")) for p in sorted(glob.glob(os.path.join(args.real_refs, "*.png")))]
        rng = np.random.default_rng(0)
        style = (lambda im: random_real_style(im, refs, rng)) if refs else None
    if args.mode == "2.5d":
        tr, va = SliceDataset(args.data, "train", args.context, args.crop, style=style), SliceDataset(args.data, "val", args.context, args.crop)
        in_ch = 2 * args.context + 1
    else:
        tr, va = PatchDataset3D(args.data, "train", tuple(args.patch)), PatchDataset3D(args.data, "val", tuple(args.patch))
        in_ch = 1
    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = build_model(args.arch, in_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl_tr))
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    os.makedirs(args.out, exist_ok=True)
    log = csv.writer(open(os.path.join(args.out, "log.csv"), "w", newline=""))
    log.writerow(["epoch", "train_loss", "val_iou", "val_dice", "val_pixacc", "val_porosity_abs_err", "sec"])
    best = -1
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tl = 0
        for batch in dl_tr:
            if args.mode == "2.5d":
                x, y, y1, sdf = [b.to(dev) for b in batch]
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    logits = model(x)
                    logits_next = None
                    if args.w_cons > 0:  # prediction for z+1 = shift the context window by one slice
                        x_next = torch.cat([x[:, 1:], x[:, -1:]], 1)
                        logits_next = model(x_next)
                    loss = total_loss(logits, y, sdf, args.w_ce, args.w_dice, args.w_bd, args.w_cons, logits_next, y1)
            else:
                x, y = [b.to(dev) for b in batch]
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    logits = model(x)
                    loss = total_loss(logits, y, None, args.w_ce, args.w_dice)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item()
        model.eval(); agg = {}
        with torch.no_grad():
            for batch in dl_va:
                x, y = batch[0].to(dev), batch[1].to(dev)
                m = metrics(model(x), y)
                for k, v in m.items():
                    agg[k] = agg.get(k, 0) + v / len(dl_va)
        log.writerow([ep, tl / len(dl_tr), agg["iou"], agg["dice"], agg["pixacc"], agg["porosity_abs_err"], time.time() - t0])
        print(f"ep {ep:3d} loss {tl/len(dl_tr):.4f} val IoU {agg['iou']:.4f} Dice {agg['dice']:.4f} porosity|err| {agg['porosity_abs_err']:.4f} ({time.time()-t0:.0f}s)")
        if agg["iou"] > best:
            best = agg["iou"]
            torch.save(dict(model=model.state_dict(), args=vars(args), val=agg), os.path.join(args.out, "best.pt"))
    print("best val IoU", best)


# --------------------------------------------------------------------------------------
# stack inference + descriptor-level evaluation (what the mentor actually cares about)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict_stack(model, images: np.ndarray, context: int = 1, dev="cpu", batch: int = 8) -> np.ndarray:
    model.eval()
    N = images.shape[0]
    out = np.zeros(images.shape, bool)
    for s in range(0, N, batch):
        xs = []
        for z in range(s, min(s + batch, N)):
            zs = [min(max(z + d, 0), N - 1) for d in range(-context, context + 1)]
            xs.append(np.stack([images[j] for j in zs]).astype(np.float32))
        x = torch.from_numpy((np.stack(xs) / 255.0 - 0.35) / 0.25).to(dev)
        out[s:s + len(xs)] = model(x).argmax(1).cpu().numpy() == 1
    return out


def evaluate_volume(pred_solid: np.ndarray, gt_solid: np.ndarray, voxel_nm: float):
    """3-D IoU + descriptor errors (porosity, PSD mu/sigma, mean pore) — the acceptance test."""
    from .descriptors import describe_volume
    inter = (pred_solid & gt_solid).sum(); union = (pred_solid | gt_solid).sum()
    dp, dg = describe_volume(pred_solid, voxel_nm, clean_radius=2), describe_volume(gt_solid, voxel_nm, clean_radius=2)
    return dict(iou3d=float(inter / max(union, 1)), porosity_pred=dp.porosity, porosity_gt=dg.porosity,
                porosity_rel_err=abs(dp.porosity - dg.porosity) / max(dg.porosity, 1e-6),
                pore_mean_pred=dp.pore_mean_nm, pore_mean_gt=dg.pore_mean_nm,
                mu_pred=dp.lognormal_mu, mu_gt=dg.lognormal_mu, sigma_pred=dp.lognormal_sigma, sigma_gt=dg.lognormal_sigma)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--out", default="runs/exp")
    p.add_argument("--arch", default="deeplabv3plus", choices=["deeplabv3plus", "unet2d", "unet3d"])
    p.add_argument("--mode", default="2.5d", choices=["2.5d", "3d"])
    p.add_argument("--context", type=int, default=1, help="slices each side (1 -> prev/cur/next)")
    p.add_argument("--crop", type=int, default=128); p.add_argument("--patch", type=int, nargs=3, default=[32, 128, 128])
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--w_ce", type=float, default=0.5); p.add_argument("--w_dice", type=float, default=0.5)
    p.add_argument("--w_bd", type=float, default=0.0); p.add_argument("--w_cons", type=float, default=0.0)
    p.add_argument("--real_refs", default=None, help="folder of real PNG crops for FDA/histogram style augmentation")
    return p


if __name__ == "__main__":
    train(get_parser().parse_args())
