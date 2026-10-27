"""Pretrain any architecture on the SliceGAN digital twin, matched to the three acquisitions.

Geometry: the 14 SliceGAN volumes (runs/slicegan/volumes.npz; SliceGAN, Kench & Cooper 2021, trained on
real comp_full cut faces). Imaging: each volume is rendered three times through the calibrated
shine-through twin of comp, uncomp and pristine (runs/twin_{stack}/calibration.json: slide rate,
cut-face / background / floor grey levels, depth decay), so the network sees exactly the three
acquisitions it will be fine-tuned on, with exact pore/solid truth. 12 volumes train, 2 validate.

    python eponge/scripts/twin_pretrain.py --arch zafunet --k 3 --out runs/twin_gan3/zafunet_k3
then: train_final.py ... --init runs/twin_gan3/zafunet_k3/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import twin_experiment as TE  # noqa: E402
from eponge import simulate as SIM  # noqa: E402
from eponge.inference import _device  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def imaging(stack):
    c = json.loads((ROOT / f"runs/twin_{stack}/calibration.json").read_text())["params"]
    return SIM.Imaging(**{k: v for k, v in c.items() if k in SIM.Imaging.__dataclass_fields__})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--learned-slide", action="store_true")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--geometry", default="comp_full", choices=["comp_full", "per_stack"],
                    help="per_stack: SliceGAN trained on each acquisition's hand masks (runs/slicegan_<stack>)")
    ap.add_argument("--gan", default=None, help="GAN-twin run folder: replace physics images by GAN-twin images")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    TE.OUT = out.parent
    z = np.load(ROOT / "runs/slicegan/volumes.npz")
    vols = [z[k] for k in sorted(z.files, key=lambda s: int(s.split("_")[1]))]
    train, val = [], []
    geo = {}
    for i, stack in enumerate(("comp", "uncomp", "pristine")):
        if a.geometry == "per_stack":
            zz = np.load(ROOT / f"runs/slicegan_{stack}/volumes.npz")
            vols = [zz[k] for k in sorted(zz.files, key=lambda s: int(s.split("_")[1]))]
        geo[stack] = vols
        p = imaging(stack)
        train += SIM.make_dataset(0, seed=100 + i, p=p, geometries=vols[:12])
        val += SIM.make_dataset(0, seed=200 + i, p=p, geometries=vols[12:14])
    if a.gan:
        import twin_refiner as TR
        dev = _device(None)
        G = TR.load_G(dev, a.gan)
        rng = np.random.default_rng(5)
        for i, stack in enumerate(("comp", "uncomp", "pristine")):
            for d, g in zip(train[12 * i:12 * (i + 1)], geo[stack][:12]):
                d["images"] = TR.gan_volume(G, g, stack, dev, rng)
            for d, g in zip(val[2 * i:2 * (i + 1)], geo[stack][12:14]):
                d["images"] = TR.gan_volume(G, g, stack, dev, rng)
        del G
        print("GAN-twin images for", len(train), "train and", len(val), "val volumes", flush=True)
    cfg = {"arch": a.arch}
    if a.arch == "zafunet":
        cfg["encoder"] = a.encoder
    if a.arch == "millnet":
        cfg.update(use_bank=True, learned_slide=a.learned_slide)
    lr = a.lr or {"zafunet": 4e-4, "unet_r34": 4e-4}.get(a.arch, 2e-3)
    TE.train_on_twin(out.name, cfg, a.k, train, val, _device(None), iters=a.iters, lr=lr)
    print("twin-pretrained", out / "best.pt", flush=True)


if __name__ == "__main__":
    main()
