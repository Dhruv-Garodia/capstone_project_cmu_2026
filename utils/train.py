#!/usr/bin/env python3
# scripts/train.py

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset

# Repo imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.unet import UNet
from model.dataset import PFIBSliceDataset
from model.transforms import ComposePair, HFlip, VFlip, RandomGamma, ToTensor


# ---------------- Utility ----------------
def set_seed(seed: int = 2025):
    """Make runs reproducible-ish."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    """Pick device with manual override (PFIB_FORCE_DEVICE), else CUDA->MPS->CPU."""
    force = os.environ.get("PFIB_FORCE_DEVICE")
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------- Transforms ----------------
def build_transforms(flip_h_p=0.5, flip_v_p=0.1, gamma_p=0.0,
                     gamma_low=0.95, gamma_high=1.05):
    """
    Light, geometry-safe augmentations:
      - Paired flips keep image/mask aligned
      - Mild gamma is optional (image-only)
    """
    both = ()
    if flip_h_p > 0:
        both += (HFlip(p=flip_h_p),)
    if flip_v_p > 0:
        both += (VFlip(p=flip_v_p),)

    image_only = ()
    if gamma_p > 0:
        image_only += (RandomGamma(gamma_range=(gamma_low, gamma_high), p=gamma_p),)

    return ComposePair(both=both, image_only=image_only), ToTensor()


# ---------------- Metrics ----------------
def accuracy_pixel(logits: torch.Tensor, mask: torch.Tensor) -> float:
    """Simple pixel accuracy."""
    pred = logits.argmax(dim=1)
    correct = (pred == mask).sum().item()
    total = mask.numel()
    return correct / max(1, total)


def iou_class1(logits: torch.Tensor, mask: torch.Tensor) -> float:
    """IoU for the positive class (label=1)."""
    pred = logits.argmax(dim=1)  # [B,H,W]
    p1 = (pred == 1)
    t1 = (mask == 1)
    inter = (p1 & t1).sum().item()
    union = (p1 | t1).sum().item()
    return inter / max(1, union)


# ---------------- Loss building blocks ----------------
# def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     """
#     Soft Dice loss on the positive class.
#       logits: [B,2,H,W]
#       target: [B,H,W] in {0,1}
#     Returns 1 - Dice(pos).
#     """
#     probs1 = F.softmax(logits, dim=1)[:, 1]  # [B,H,W]
#     tgt = target.float()
#     inter = (probs1 * tgt).sum(dim=(1, 2))
#     union = probs1.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2)) + eps
#     dice = (2 * inter + eps) / union
#     return 1.0 - dice.mean()

def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Symmetric Soft Dice loss over BOTH classes (0: background, 1: foreground).
      logits: [B,2,H,W]
      target: [B,H,W] in {0,1}
    Returns 1 - 0.5*(Dice(bg) + Dice(fg)).
    """
    probs = F.softmax(logits, dim=1)        # [B,2,H,W]
    p1 = probs[:, 1]                        # foreground prob
    p0 = probs[:, 0]                        # background prob

    t = target.float()                      # [B,H,W]
    t1 = t                                  # foreground gt
    t0 = 1.0 - t                            # background gt

    # Dice for a single class
    def _dice(p, g):
        inter = (p * g).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2)) + eps
        return (2.0 * inter + eps) / denom   # [B]

    dice1 = _dice(p1, t1)                   # foreground dice
    dice0 = _dice(p0, t0)                   # background dice

    dice_sym = 0.5 * (dice1 + dice0)       # [B]
    return 1.0 - dice_sym.mean()

def tversky_loss(logits: torch.Tensor, target: torch.Tensor,
                 alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
    """
    Tversky loss (generalized Dice).
    alpha weights FP, beta weights FN. beta > alpha => more penalty on FN (under-segmentation).
    """
    probs = F.softmax(logits, dim=1)[:, 1]       # foreground prob [B,H,W]
    t = target.float()                           # [B,H,W]

    # TP, FP, FN
    TP = (probs * t).sum(dim=(1, 2))
    FP = (probs * (1.0 - t)).sum(dim=(1, 2))
    FN = ((1.0 - probs) * t).sum(dim=(1, 2))

    tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    return 1.0 - tversky.mean()

def focal_tversky_loss(logits: torch.Tensor, target: torch.Tensor,
                       alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Focal Tversky Loss.
    gamma > 1 amplifies focus on hard examples.
    """
    probs = F.softmax(logits, dim=1)[:, 1]
    t = target.float()

    TP = (probs * t).sum(dim=(1, 2))
    FP = (probs * (1.0 - t)).sum(dim=(1, 2))
    FN = ((1.0 - probs) * t).sum(dim=(1, 2))

    tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    focal_ti = (1.0 - tversky).pow(gamma)
    return focal_ti.mean()

def compute_class_weights(dataloader: DataLoader, max_batches: int = 10) -> torch.Tensor:
    """
    Estimate class weights for CE by pixel frequency on a few batches.
    Returns a tensor [2] for classes {0,1}, inversely proportional to frequency.
    """
    n0 = n1 = 0
    with torch.no_grad():
        for bidx, (_, m) in enumerate(dataloader):
            n0 += (m == 0).sum().item()
            n1 += (m == 1).sum().item()
            if bidx + 1 >= max_batches:
                break
    total = max(1, n0 + n1)
    f0, f1 = n0 / total, n1 / total
    # inverse frequency then normalize
    w0 = 1.0 / max(1e-6, f0)
    w1 = 1.0 / max(1e-6, f1)
    s = w0 + w1
    return torch.tensor([w0 / s, w1 / s], dtype=torch.float32)


def focal_loss_softmax(logits: torch.Tensor, target: torch.Tensor,
                       gamma: float = 2.0, alpha: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    Simple focal loss for 2-class softmax.
    alpha weights the positive class; gamma controls focusing on hard examples.
    """
    probs = F.softmax(logits, dim=1)  # [B,2,H,W]
    p1 = probs[:, 1]                  # [B,H,W]
    t = target.float()
    # positive term
    pos = -alpha * (1 - p1).pow(gamma) * (t * (p1 + eps).log())
    # negative term
    p0 = probs[:, 0]
    neg = -(1 - alpha) * (1 - p0).pow(gamma) * ((1 - t) * (p0 + eps).log())
    return (pos + neg).mean()


# ---------------- Main Training ----------------
def main():
    parser = argparse.ArgumentParser(description="PFIB-SEM Step 1: 2D Slice Segmentation")

    # Data / I/O
    parser.add_argument("--img_dir", type=str, required=True, help="Input 2D slice directory (lightened images)")
    parser.add_argument("--mask_dir", type=str, required=True, help="Binary mask directory (1=solid, 0=pore)")
    parser.add_argument("--out", type=str, default="checkpoints")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Optional subdirectory name under --out for this run")

    # Train config
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default=None,
                        help="Optional manual device override, e.g. cpu, cuda, mps")

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"],
                        help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay (set 0 for Adam if you prefer)")

    # Loss options
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "cedice", "focal", "focaldice", "tversky", "focaltversky"],
                        help="Loss type: ce, cedice, focal, focaldice, tversky, focaltversky")
    parser.add_argument("--dice_w", type=float, default=0.5,
                        help="Weight for Dice in CE+Dice: total = (1-dice_w)*CE + dice_w*Dice")
    parser.add_argument("--use_class_weight", action="store_true",
                        help="Enable class weights for CE (computed from pixel frequencies)")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--focal_alpha", type=float, default=0.5, help="Focal loss alpha for positive class")
    parser.add_argument("--tversky_alpha", type=float, default=0.3, help="Tversky FP weight alpha")
    parser.add_argument("--tversky_beta", type=float, default=0.7, help="Tversky FN weight beta")
    parser.add_argument("--tversky_gamma", type=float, default=1.5, help="Focal Tversky gamma")
    parser.add_argument("--save_by", type=str, default="pixacc", choices=["pixacc", "iou"],
                        help="Metric used to select the best checkpoint")

    # Debug: overfit one
    parser.add_argument("--overfit_one", action="store_true",
                        help="Train on a single sample to sanity-check the pipeline")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of the sample when --overfit_one is set")

    # Augmentations
    parser.add_argument("--flip_h_p", type=float, default=0.5)
    parser.add_argument("--flip_v_p", type=float, default=0.1)
    parser.add_argument("--gamma_p", type=float, default=0.0)
    parser.add_argument("--gamma_low", type=float, default=0.95)
    parser.add_argument("--gamma_high", type=float, default=1.05)

    args = parser.parse_args()

    # Setup
    out_dir = Path(args.out)
    if args.experiment_name:
        out_dir = out_dir / args.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.device:
        os.environ["PFIB_FORCE_DEVICE"] = args.device
    device = pick_device()
    print(f"[Info] Using device: {device}")

    # Transforms (disable randomness for overfit-one)
    if args.overfit_one:
        pair_tfms, to_tensor = build_transforms(flip_h_p=0.0, flip_v_p=0.0, gamma_p=0.0)
    else:
        pair_tfms, to_tensor = build_transforms(
            flip_h_p=args.flip_h_p, flip_v_p=args.flip_v_p,
            gamma_p=args.gamma_p, gamma_low=args.gamma_low, gamma_high=args.gamma_high
        )

    # Dataset
    ds = PFIBSliceDataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        transform_pair=pair_tfms,
        to_tensor=to_tensor,
        grayscale=(args.in_channels == 1),
    )

    # Split / Loaders
    if args.overfit_one:
        idx = max(0, min(args.sample_idx, len(ds) - 1))
        train_ds = Subset(ds, [idx])
        val_ds = Subset(ds, [idx])
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True,
            num_workers=0, pin_memory=False, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False
        )
        print(f"[Debug] Overfitting a single sample at index {idx}")
    else:
        val_len = max(1, int(len(ds) * args.val_split))
        train_len = len(ds) - val_len
        train_ds, val_ds = random_split(ds, [train_len, val_len])

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda")
        )

    # Model
    model = UNet(in_channels=args.in_channels, num_classes=2, base_ch=args.base_ch, bilinear=True).to(device)

    # Build loss
    if args.use_class_weight:
        _tmp_loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        class_w = compute_class_weights(_tmp_loader, max_batches=10).to(device)
        print("[Info] CE class weights:", class_w.tolist())
    else:
        class_w = None

    def ce_loss_fn(logits, masks):
        return F.cross_entropy(logits, masks, weight=class_w)

    def total_loss(logits, masks):
        if args.loss == "ce":
            return ce_loss_fn(logits, masks)

        elif args.loss == "cedice":
            ce = ce_loss_fn(logits, masks)
            dice = soft_dice_loss(logits, masks)
            return (1.0 - args.dice_w) * ce + args.dice_w * dice

        elif args.loss == "focal":
            return focal_loss_softmax(logits, masks,
                                    gamma=args.focal_gamma,
                                    alpha=args.focal_alpha)

        elif args.loss == "focaldice":
            focal = focal_loss_softmax(logits, masks,
                                    gamma=args.focal_gamma,
                                    alpha=args.focal_alpha)
            dice = soft_dice_loss(logits, masks)
            return (1.0 - args.dice_w) * focal + args.dice_w * dice

        elif args.loss == "tversky":
            return tversky_loss(logits, masks, alpha=args.tversky_alpha, beta=args.tversky_beta)

        elif args.loss == "focaltversky":
            return focal_tversky_loss(
                logits, masks,
                alpha=args.tversky_alpha,
                beta=args.tversky_beta,
                gamma=args.tversky_gamma,
            )

        else:
            raise ValueError(f"Unknown --loss {args.loss}")


    # Optimizer & Scheduler
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:  # adamw
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    tmax = max(10, args.epochs) if args.overfit_one else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)

    # Training loop
    best_val_acc = 0.0
    best_val_iou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)
            loss = total_loss(logits, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_acc_sum, val_iou_sum, val_n = 0.0, 0.0, 0
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                val_acc_sum += accuracy_pixel(logits, masks) * masks.size(0)
                val_iou_sum += iou_class1(logits, masks) * masks.size(0)
                val_n += masks.size(0)
        val_acc = val_acc_sum / max(1, val_n)
        val_iou = val_iou_sum / max(1, val_n)
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(loss_sum / len(train_loader)),
            "val_pixAcc": float(val_acc),
            "val_IoU_pos": float(val_iou),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)

        print(f"[Epoch {epoch:03d}] "
              f"train_loss={epoch_record['train_loss']:.4f}  "
              f"val_pixAcc={val_acc:.4f}  val_IoU(1)={val_iou:.4f}")

        current_score = val_acc if args.save_by == "pixacc" else val_iou
        best_score = best_val_acc if args.save_by == "pixacc" else best_val_iou

        if current_score > best_score:
            best_val_acc = max(best_val_acc, val_acc)
            best_val_iou = max(best_val_iou, val_iou)
            ckpt = out_dir / "unet_best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_pixAcc": val_acc, "val_IoU_pos": val_iou}, ckpt)
            print(f"  ✓ Saved best model ({args.save_by}={current_score:.4f})")

        # Optional early stop for overfit_one when it's essentially solved
        if args.overfit_one and val_acc >= 0.99:
            print("  ✓ Reached ~100% pixel accuracy on the single sample. Stopping early.")
            break

    # Save last checkpoint
    last_ckpt = out_dir / "unet_last.pt"
    torch.save({"model": model.state_dict(), "epoch": epoch}, last_ckpt)

    history_json = out_dir / "history.json"
    metrics_csv = out_dir / "history.csv"
    summary_json = out_dir / "summary.json"

    with history_json.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_pixAcc", "val_IoU_pos", "lr"])
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "img_dir": args.img_dir,
        "mask_dir": args.mask_dir,
        "out_dir": str(out_dir),
        "loss": args.loss,
        "optimizer": args.optimizer,
        "save_by": args.save_by,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "base_ch": args.base_ch,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "tversky_alpha": args.tversky_alpha,
        "tversky_beta": args.tversky_beta,
        "tversky_gamma": args.tversky_gamma,
        "best_val_pixAcc": best_val_acc,
        "best_val_IoU_pos": best_val_iou,
        "best_checkpoint": str(out_dir / "unet_best.pt"),
        "last_checkpoint": str(last_ckpt),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Last checkpoint: {last_ckpt}")
    print(f"[Info] Saved run history to: {history_json}")
    print(f"[Info] Saved run summary to: {summary_json}")


if __name__ == "__main__":
    main()
