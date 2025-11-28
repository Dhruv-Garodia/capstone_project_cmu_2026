#!/usr/bin/env python3
"""
PFIB-SEM inference (full image, minimal I/O).

- Input : single image (--img) or folder (--img_dir) of lightened grayscale slices
- Output: predicted binary masks (PNG)
- Model : UNet saved .pt (same arch as model/unet.py)

This script:
* feeds the whole image into UNet (no sliding windows),
* pads to multiples of 16 on bottom/right, then crops back,
* outputs argmax masks (0/1) only.
"""

import sys
import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# import UNet from project
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from model.unet import UNet

def dilate_pore(pred, radius=1):
    # pred: [B,1,H,W] float 0/1 (0=pore, 1=solid)
    inv = 1 - pred                                   # 反转，让 pore 变成 1
    dil = F.max_pool2d(inv, kernel_size=2*radius+1,
                       stride=1, padding=radius)    # 普通 dilation
    return 1 - dil                                   # 再反转回来

def pick_device(force: str | None = None) -> torch.device:
    """Pick device with optional override."""
    if force is not None:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pad_to_16(img_t: torch.Tensor):
    """Pad [1,1,H,W] to multiples of 16 on bottom/right; return padded and original (H,W)."""
    _, _, H, W = img_t.shape
    H2 = (H + 15) // 16 * 16
    W2 = (W + 15) // 16 * 16
    pad_h, pad_w = H2 - H, W2 - W
    if pad_h or pad_w:
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h))
    return img_t, (H, W)


def list_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


# def infer_one(model: torch.nn.Module, device: torch.device, pil_img: Image.Image):
#     """Full-image inference -> binary mask [1,1,H,W] float {0,1}."""
#     model.eval()
#     with torch.no_grad():
#         x = to_tensor(pil_img.convert("L")).unsqueeze(0).to(device)   # [1,1,H,W]
#         logits = model(x)                                             # [1,2,H,W]
#         pred = logits.argmax(dim=1, keepdim=True).float()             # [1,1,H,W]
#         return pred

def infer_one(model: torch.nn.Module, device: torch.device, pil_img: Image.Image, thresh: float = 0.5):
    """Full-image inference -> binary mask [1,1,H,W] float {0,1}."""
    model.eval()
    with torch.no_grad():
        x = to_tensor(pil_img.convert("L")).unsqueeze(0).to(device)   # [1,1,H,W]                      # [1,2,H,W]
        logits = model(x)
        # ← 新增：softmax 得到 foreground 概率
        probs = F.softmax(logits, dim=1)[:, 1:2]                      # [1,1,H,W]
        # ← 用你传进来的阈值
        # 大于thresh是白色
        pred = (probs > thresh).float()                               # [1,1,H,W]
        # print("thresh =", thresh)
        # print("probs min/max/mean:", probs.min().item(), probs.max().item(), probs.mean().item())
        # for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        #     frac = (probs > t).float().mean().item()
        #     print(f"  fraction of pixels with prob > {t}: {frac:.4f}")
        # pred = dilate_pore(pred, radius=1)
        return pred


def main():
    ap = argparse.ArgumentParser("PFIB minimal inference (full image)")
    # Inputs
    ap.add_argument("--img", type=str, help="single image file")
    ap.add_argument("--img_dir", type=str, help="folder with images (recursive)")
    ap.add_argument("--out_dir", required=True, help="where to save predicted masks")
    # Model
    ap.add_argument("--ckpt", required=True, help="path to .pt model")
    ap.add_argument("--in_channels", type=int, default=1)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--bilinear", action="store_true", default=True)
    # Device
    ap.add_argument("--thresh", type=float, default=0.45, help="probability threshold for pore (foreground)")
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    args = ap.parse_args()

    device = pick_device(args.device)
    print("[infer] using device:", device)

    # (macOS OpenMP stability; harmless elsewhere)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Model
    model = UNet(in_channels=args.in_channels, num_classes=2,
                 base_ch=args.base_ch, bilinear=args.bilinear).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    print(f"[infer] loaded model from {args.ckpt}")

    # Files
    if args.img and args.img_dir:
        raise ValueError("Pass either --img or --img_dir, not both.")
    if args.img:
        img_list = [Path(args.img)]
        root_for_rel = Path(args.img).parent
    elif args.img_dir:
        root_for_rel = Path(args.img_dir)
        img_list = list_images(root_for_rel)
        if not img_list:
            raise FileNotFoundError(f"No images found under {root_for_rel}")
    else:
        raise ValueError("You must pass --img or --img_dir")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Run
    with torch.no_grad():
        for p in img_list:
            try:
                with Image.open(p) as im:
                        img = im.convert("L")
            except OSError as e:
                print(f"[WARN] skipping corrupted image {p}: {e}")
                continue
            pred = infer_one(model, device, img, thresh=args.thresh)                # [1,1,H,W]
            out_path = out_root / p.relative_to(root_for_rel)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(pred, out_path)                          # saves {0,1} mask
            print(f"[infer] saved {out_path}")

    print("[infer] done.")


if __name__ == "__main__":
    main()
