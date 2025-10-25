# scripts/train.py
import argparse, os, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model.unet import UNet
from model.dataset import PFIBSliceDataset
from model.transforms import ComposePair, HFlip, VFlip, RandomGamma, ToTensor


# ---------------- Utility ----------------
def set_seed(seed: int = 2025):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------- Transforms ----------------
def build_transforms(flip_h_p=0.5, flip_v_p=0.1, gamma_p=0.0,
                     gamma_low=0.95, gamma_high=1.05):
    """
    Light, geometry-safe augmentation for 'perfect' inputs.
    - Paired flips keep mask alignment
    - Mild gamma optional
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


def accuracy_pixel(pred_logits: torch.Tensor, mask: torch.Tensor) -> float:
    pred = pred_logits.argmax(dim=1)
    correct = (pred == mask).sum().item()
    total = mask.numel()
    return correct / max(1, total)


# ---------------- Main Training ----------------
def main():
    parser = argparse.ArgumentParser(description="PFIB-SEM Step 1: 2D Slice Segmentation")
    parser.add_argument("--img_dir", type=str, required=True, help="Input 2D slice directory (lightened images)")
    parser.add_argument("--mask_dir", type=str, required=True, help="Binary mask directory (1=solid, 0=pore)")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=2025)
    # light augment knobs
    parser.add_argument("--flip_h_p", type=float, default=0.5)
    parser.add_argument("--flip_v_p", type=float, default=0.1)
    parser.add_argument("--gamma_p", type=float, default=0.0)
    parser.add_argument("--gamma_low", type=float, default=0.95)
    parser.add_argument("--gamma_high", type=float, default=1.05)

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = pick_device()
    print(f"[Info] Using device: {device}")

    # ---------- dataset ----------
    pair_tfms, to_tensor = build_transforms(
        flip_h_p=args.flip_h_p, flip_v_p=args.flip_v_p,
        gamma_p=args.gamma_p, gamma_low=args.gamma_low, gamma_high=args.gamma_high
    )

    ds = PFIBSliceDataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        transform_pair=pair_tfms,
        to_tensor=to_tensor,
        grayscale=(args.in_channels == 1),
    )

    # split train/val (keep sequential, not random shuffle)
    val_len = max(1, int(len(ds) * args.val_split))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    # ------------- loaders -------------
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,          # Note: shuffle=False to preserve z-order continuity.
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ------------- model -------------
    model = UNet(in_channels=args.in_channels, num_classes=2, base_ch=32, bilinear=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        # -------- validation --------
        model.eval()
        with torch.no_grad():
            val_acc_sum, val_n = 0.0, 0
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                val_acc_sum += accuracy_pixel(logits, masks) * masks.size(0)
                val_n += masks.size(0)
        val_acc = val_acc_sum / max(1, val_n)
        scheduler.step()

        print(f"[Epoch {epoch:03d}] train_loss={loss_sum/len(train_loader):.4f}  val_pixAcc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = os.path.join(args.out, "unet_best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_pixAcc": val_acc}, ckpt)
            print(f"  ✓ Saved best model (pixAcc={val_acc:.4f})")

    last_ckpt = os.path.join(args.out, "unet_last.pt")
    torch.save({"model": model.state_dict(), "epoch": args.epochs}, last_ckpt)
    print(f"Done. Last checkpoint: {last_ckpt}")


if __name__ == "__main__":
    main()
