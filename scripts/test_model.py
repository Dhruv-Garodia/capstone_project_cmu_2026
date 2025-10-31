#!/usr/bin/env python3
"""
Simple PFIB-SEM inference script.
Input:  lightened slices folder
Output: predicted binary masks (PNG)
Model:  saved UNet .pt (same arch as model/unet.py)
"""
import sys
import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from PIL import Image

# import your UNet
# assumes your repo has model/unet.py
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from model.unet import UNet


def pad_to_16(t):
    """Pad tensor [1,1,H,W] to multiples of 16 on bottom/right."""
    _, _, h, w = t.shape
    nh = (h + 15) // 16 * 16
    nw = (w + 15) // 16 * 16
    pad_h = nh - h
    pad_w = nw - w
    if pad_h == 0 and pad_w == 0:
        return t, (h, w)
    t = F.pad(t, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)
    return t, (h, w)


def main():
    ap = argparse.ArgumentParser("PFIB simple inference")
    ap.add_argument("--img_dir", required=True, help="folder with lightened slices")
    ap.add_argument("--ckpt", required=True, help="path to .pt model")
    ap.add_argument("--out_dir", required=True, help="where to save predicted masks")
    args = ap.parse_args()

    img_root = Path(args.img_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print("[infer] using device:", device)

    # load model
    model = UNet(in_channels=1, num_classes=2, base_ch=32, bilinear=True).to(device)
    state = torch.load(args.ckpt, map_location=device)
    # support both {"model": ...} and plain state_dict
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"[infer] loaded model from {args.ckpt}")

    # scan images recursively
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    img_files = [
        p for p in img_root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
    if not img_files:
        raise FileNotFoundError(f"No images found under {img_root}")

    with torch.no_grad():
        for img_path in img_files:
            # load image (grayscale)
            img = Image.open(img_path).convert("L")
            img_t = to_tensor(img).unsqueeze(0).to(device)  # [1,1,H,W]

            # pad to 16 so UNet is happy
            img_t_pad, orig_hw = pad_to_16(img_t)

            # forward
            logits = model(img_t_pad)          # [1,2,H',W']
            preds = torch.argmax(logits, 1)    # [1,H',W']

            # crop back to original
            h, w = orig_hw
            pred = preds[:, :h, :w]            # [1,H,W]

            # save to out_dir with same rel path
            rel = img_path.relative_to(img_root)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # pred is {0,1} -> make it float for saving
            save_image(pred.float(), out_path)
            print(f"[infer] saved {out_path}")

    print("[infer] done.")


if __name__ == "__main__":
    # avoid mac openmp crash for this script
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
