"""Export the digital-twin dataset: simulated pFIB-SEM stacks with exact pore/solid masks.

Geometry: 14 SliceGAN volumes (trained on real comp_full cut faces), 6 real-derived volumes cut from a
segmentation of comp_full, and 6 hand-label volumes per stack built from the final annotated masks (paired with
the real images of the same crop). Imaging: each geometry rendered through the calibrated twin of each acquisition
(comp, uncomp, pristine: slide rate, grey levels, shine-through depth decay, detector noise and curtaining).

    python eponge/scripts/export_twin_dataset.py      # -> ../twin_dataset/
"""
import json
from pathlib import Path

import numpy as np
import tifffile

from eponge import simulate as SIM

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent / "twin_dataset"


FINAL = ROOT.parent / "final_dataset"


def hand_label_volumes(stack, n=6, depth=64, width=256, seed=21):
    """3D blocks of the team's hand-annotated masks (slices 0-117 stacked), with the real images of the same crop.

    Returns [(solid, real_images, crop)] where crop = {z0, y0, y1, x0, x1}; missing label slices take the
    nearest labelled neighbour."""
    from PIL import Image
    from eponge.metrics import largest_inner_rectangle
    images = tifffile.imread(FINAL / stack / "images.tif")
    files = {int(f.stem.split("_")[1]): f for f in (FINAL / stack / "masks").glob("slice_*.png")}
    zs = sorted(files)
    m = {z: np.array(Image.open(files[z])) for z in zs}
    cls = np.stack([np.select([m[z] == 255, m[z] == 0], [0, 1], 2).astype(np.uint8)
                    for z in [min(zs, key=lambda q: abs(q - t)) for t in range(zs[0], zs[-1] + 1)]])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n * 4):
        if len(out) == n:
            break
        z0 = int(rng.integers(0, len(cls) - depth + 1))
        rs, cs = largest_inner_rectangle((cls[z0:z0 + depth] != 2).all(0))
        if cs.stop - cs.start < width or rs.stop - rs.start < 48:
            continue
        x0 = int(rng.integers(cs.start, cs.stop - width + 1))
        crop = {"z0": z0, "y0": rs.start, "y1": rs.stop, "x0": x0, "x1": x0 + width}
        solid = np.ascontiguousarray(cls[z0:z0 + depth, rs, x0:x0 + width] == 0)
        real = images[zs[0] + z0:zs[0] + z0 + depth, rs, x0:x0 + width]
        if any(c["z0"] == z0 and c["x0"] == x0 for _, _, c in out):
            continue
        out.append((solid, real, crop))
    return out


def main():
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan", default=None, help="GAN-twin run folder: also write images_gan.tif for every volume")
    ap.add_argument("--geometry", default="comp_full", choices=["comp_full", "per_stack"])
    a = ap.parse_args()
    G = None
    if a.gan:
        sys.path.insert(0, str(Path(__file__).parent))
        import twin_refiner as TR
        from eponge.inference import _device
        dev = _device(None)
        G = TR.load_G(dev, a.gan)
        grng = np.random.default_rng(13)
    z = np.load(ROOT / "runs/slicegan/volumes.npz")
    gan = [z[k] for k in sorted(z.files, key=lambda s: int(s.split("_")[1]))]
    cls = np.load(ROOT / "runs/twin/comp_full_millnet_classes.npy")
    real = SIM.real_geometry(cls, range(0, 354), 6, depth=64, seed=11)
    index = []
    for stack in ("comp", "uncomp", "pristine"):
        cal = json.loads((ROOT / f"runs/twin_{stack}/calibration.json").read_text())
        (OUT / stack).mkdir(parents=True, exist_ok=True)
        (OUT / stack / "calibration.json").write_text(json.dumps(cal, indent=1))
        c = cal["params"]
        p = SIM.Imaging(**{k: v for k, v in c.items() if k in SIM.Imaging.__dataclass_fields__})
        if a.geometry == "per_stack":
            zz = np.load(ROOT / f"runs/slicegan_{stack}/volumes.npz")
            gan = [zz[k] for k in sorted(zz.files, key=lambda s: int(s.split("_")[1]))]
        geos = [("slicegan", i, g) for i, g in enumerate(gan)] + \
               ([] if a.geometry == "per_stack" else [("real", i, g) for i, g in enumerate(real)])   # comp_full blocks: wrong scale for per-stack
        rendered = SIM.make_dataset(0, seed=500, p=p, geometries=[g for _, _, g in geos])
        for (src, i, g), d in zip(geos, rendered):
            name = f"{stack}_{src}_{i:02d}"
            dd = OUT / stack / name
            dd.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(dd / "images.tif", d["images"].astype(np.uint8), compression="zlib")
            if G is not None:
                tifffile.imwrite(dd / "images_gan.tif", TR.gan_volume(G, g, stack, dev, grng), compression="zlib")
            tifffile.imwrite(dd / "masks.tif", np.where(d["truth"], 0, 255).astype(np.uint8), compression="zlib")
            meta = {"name": name, "acquisition": stack, "geometry": src, "shape": list(d["images"].shape),
                    "porosity": float(d["truth"].mean()), "imaging": d["imaging"]}
            (dd / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
            index.append(meta)
        # hand-label geometry: the team's final masks as 3D structure, rendered through this stack's twin,
        # stored with the real images of the same crop (paired real vs twin)
        hl = hand_label_volumes(stack)
        rendered = SIM.make_dataset(0, seed=700, p=p, geometries=[g for g, _, _ in hl])
        for i, ((g, real_img, crop), d) in enumerate(zip(hl, rendered)):
            name = f"{stack}_handlabel_{i:02d}"
            dd = OUT / stack / name
            dd.mkdir(parents=True, exist_ok=True)
            n_r = len(d["images"])
            real_img = real_img[:n_r]
            tifffile.imwrite(dd / "images.tif", d["images"].astype(np.uint8), compression="zlib")
            tifffile.imwrite(dd / "masks.tif", np.where(d["truth"], 0, 255).astype(np.uint8), compression="zlib")
            tifffile.imwrite(dd / "real_images.tif", real_img.astype(np.uint8), compression="zlib")
            if G is not None:
                tifffile.imwrite(dd / "images_gan.tif", TR.gan_volume(G, g, stack, dev, grng), compression="zlib")
            r = float(np.corrcoef(d["images"].reshape(n_r, -1).astype(float).ravel(), real_img.reshape(n_r, -1).astype(float).ravel())[0, 1])
            pct = [5, 25, 50, 75, 95]
            meta = {"name": name, "acquisition": stack, "geometry": "handlabel", "shape": list(d["images"].shape),
                    "crop_in_final_dataset": {**crop, "slices": [crop["z0"], crop["z0"] + n_r - 1]},
                    "porosity": float(d["truth"].mean()), "imaging": d["imaging"],
                    "paired_pixel_correlation_twin_vs_real": r,
                    "grey_percentiles_twin": np.percentile(d["images"], pct).tolist(),
                    "grey_percentiles_real": np.percentile(real_img, pct).tolist()}
            (dd / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
            index.append(meta)
            print(name, "porosity", round(meta["porosity"], 3), "paired r", round(r, 3), flush=True)
    (OUT / "index.json").write_text(json.dumps(index, indent=1, default=float))
    (OUT / "README.md").write_text(
        "# Digital-twin dataset (simulated pFIB-SEM with exact truth)\n\n"
        "Each folder: `images.tif` (slices x H x W, uint8, physics-twin images), `images_gan.tif` (GAN-twin images: the "
        "appearance learned from the real stacks, same structure and planes) and `masks.tif` "
        "(0 = pore, 255 = solid, exact: the phase of each cut plane). `meta.json` gives the geometry source, "
        "true porosity and the imaging parameters.\n\n"
        "* Geometry: 14 SliceGAN volumes (Kench & Cooper 2021, trained on real comp_full cut faces) and 6 volumes "
        "cut from a segmentation of comp_full.\n"
        "* Imaging: rendered through the calibrated twin of each acquisition (comp: 1.2 px/slice shine-through slide; "
        "uncomp: 1.15; pristine: ~0, lower contrast). Pores show material behind the cut plane, brightness decaying "
        "with depth; plus PSF blur, detector noise, curtaining and grain.\n\n"
        "Unlike the real dataset, these masks are exact, so they measure how well a method handles shine-through.\n")
    print(len(index), "volumes ->", OUT)


if __name__ == "__main__":
    main()
