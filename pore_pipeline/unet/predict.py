"""Segment a whole stack with a trained 2.5D U-Net.

    python -m pore_pipeline.unet.predict data/comp_full.tif outputs/comp \
        --ckpt outputs/comp/unet_k3/best.pt --out outputs/comp/unet_k3/masks

The catalyst-layer region is the run's own ROI (consensus mask); pass --region to use
hand-drawn borders where they exist. Writes slice_XXXX.png masks and porosity.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .. import labels as L
from .. import preprocess as P
from .data import pad_to, window
from .model import UNet


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_model(ckpt, device):
    state = torch.load(ckpt, map_location=device)
    model = UNet(2 * state["k"] + 1, state["base"], state["depth"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval(), state["k"]


@torch.no_grad()
def pore_probability(model, frames, zs, k, device, batch=2):
    """Yield (z, probability map HxW float32) for each requested slice."""
    for i in range(0, len(zs), batch):
        chunk = zs[i:i + batch]
        x = torch.from_numpy(np.stack([window(frames, z, k) for z in chunk]).astype(np.float32) / 255.0).to(device)
        x, (h, w) = pad_to(x, model.stride)
        prob = torch.softmax(model(x), 1)[:, 1, :h, :w].cpu().numpy()
        yield from zip(chunk, prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif")
    ap.add_argument("run")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--region", help="folder of masks whose non-128 pixels define the region, where present")
    args = ap.parse_args()

    device = pick_device()
    model, k = load_model(args.ckpt, device)
    frames = P.build_cache(args.tif, args.run)
    region_files = L.slice_files(args.region) if args.region else {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for z, prob in pore_probability(model, frames, list(range(len(frames))), k, device):
        source = Path(args.region) if z in region_files else Path(args.run) / "masks" / "consensus"
        region = L.read_mask(source, z) != L.EXCLUDED
        mask = np.full(prob.shape, L.EXCLUDED, np.uint8)
        mask[region & (prob >= 0.5)] = L.PORE
        mask[region & (prob < 0.5)] = L.SOLID
        L.write_mask(out, z, mask)
        rows.append({"slice": z, "porosity": L.porosity(mask), "mean_pore_probability": float(prob[region].mean())})
    with open(out.parent / "porosity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    p = np.array([r["porosity"] for r in rows])
    print(f"{len(rows)} slices -> {out}  porosity mean {p.mean():.3f} sd {p.std():.3f}")


if __name__ == "__main__":
    main()
