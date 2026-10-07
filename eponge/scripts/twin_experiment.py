"""Digital-twin benchmark: which method recovers the true cut plane under shine-through?

Part A (exact truth). Synthetic pFIB-SEM stacks from eponge.simulate, calibrated to comp_full.
Methods: denoised Otsu, Ferner-style z-gradient method (pore_pipeline, after Ferner et al. 2024),
networks trained on the real comp_full labels, and networks trained on the twin's exact truth
(ResUNet 2D, MillNet 2D, MillNet 7-slice with and without the streak bank).

Part B (real data, label-free). The same methods on the comp_full catalyst-layer interior:
porosity (vs Ferner et al. 2024: 0.417), x-z anisotropy of the two-point function and
porosity flatness along z. The catalyst layer is in-plane isotropic, so a correct segmentation
should show no x-z difference; shine-through stretches pores along z.

    python eponge/scripts/twin_experiment.py            # everything (~1.5-2 h on an M4)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eponge import simulate as SIM  # noqa: E402
from eponge.data import CropDataset  # noqa: E402
from eponge.inference import _device, segment_stack  # noqa: E402
from eponge.io import export_to_classes, load_masks, load_stack  # noqa: E402
from eponge.models import build_model, count_parameters, load_checkpoint  # noqa: E402
from eponge.train import dice_ce_loss  # noqa: E402
from pore_pipeline import segment as GS  # noqa: E402

OUT = ROOT / "runs/twin"   # overridden by --out
R_MAX = 30


# ----------------------------------------------------------------------------- metrics
def two_point(p, axis, r_max=R_MAX):
    p = p.astype(np.float32)
    r_max = min(r_max, p.shape[axis] - 1)
    s = np.empty(r_max + 1)
    for r in range(r_max + 1):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(0, p.shape[axis] - r), slice(r, None)
        s[r] = (p[tuple(a)] * p[tuple(b)]).mean()
    return s


def anisotropy(pore):
    """x-z anisotropy of the normalised two-point function (Sardhara et al. 2022; pore_pipeline.evaluate)."""
    sx, sz = two_point(pore, 2), two_point(pore, 0)
    phi = sx[0]
    n = min(len(sx), len(sz))
    nx, nz = (sx[:n] - phi ** 2) / max(phi - phi ** 2, 1e-9), (sz[:n] - phi ** 2) / max(phi - phi ** 2, 1e-9)
    area = lambda s: float(np.clip(s, 0, None).sum() - 0.5 * (s[0] + max(s[-1], 0)))
    return {"xz_anisotropy": float(np.linalg.norm(nx - nz) / np.linalg.norm(nx)),
            "corr_len_x": area(nx), "corr_len_z": area(nz)}


def score(pred, truth):
    """pred, truth: (Z, Y, X) bool pore."""
    inter, union = (pred & truth).sum(), (pred | truth).sum()
    por_p, por_t = pred.mean((1, 2)), truth.mean((1, 2))
    return {"pore_iou": float(inter / union), "accuracy": float((pred == truth).mean()),
            "porosity_pred": float(pred.mean()), "porosity_true": float(truth.mean()),
            "porosity_bias": float(pred.mean() - truth.mean()), "porosity_mae": float(np.abs(por_p - por_t).mean()),
            **{k + "_pred": v for k, v in anisotropy(pred).items()}}


# ----------------------------------------------------------------------------- methods
def otsu_denoised(stack):
    out = []
    for f in stack:
        s = ndi.gaussian_filter(f.astype(np.float32), 1.0)
        out.append(s < threshold_otsu(s))
    return np.stack(out)


def ferner_style(stack, shift):
    """pore_pipeline's approximation of Ferner et al. 2024 on a (Z, Y, X) uint8 stack (whole frame = layer)."""
    roi = np.ones(stack.shape[1:], bool)
    sm = np.stack([GS.smooth(f) for f in stack])
    zs = list(range(0, len(stack) - 1, max(1, len(stack) // 12)))
    cal = GS.calibrate([stack[z] for z in zs], [roi] * len(zs))
    noise = GS.gradient_noise([(stack[z], stack[z + 1]) for z in zs], [roi] * len(zs), shift)
    return ~GS.ferner_style(sm, cal, shift, GS.GRAD_K * noise)


def predict(model, ck, stack, dev):
    cls, _ = segment_stack(model, stack, ck.get("k", 0), dev, tta=True)
    return cls == 1


# ----------------------------------------------------------------------------- training on the twin
def train_on_twin(name, cfg, k, train_vols, val_vols, dev, iters=3000, bs=8, lr=2e-3, seed=0, init=None):
    path = OUT / name / "best.pt"
    if path.exists():
        return load_checkpoint(path, dev)
    torch.manual_seed(seed); np.random.seed(seed)
    (OUT / name).mkdir(parents=True, exist_ok=True)
    stats = cfg["arch"] == "millnet" and cfg.get("use_stats", True)
    sets = []
    # crops must fit inside every volume (real-geometry blocks are only ~130 px tall), else they get upscaled
    crop = min(256, min(min(v["images"].shape[1:]) for v in train_vols) // 32 * 32)
    for i, v in enumerate(train_vols):
        labels = {z: v["truth"][z].astype(np.uint8) for z in range(len(v["truth"]))}   # 0 solid, 1 pore
        frames = list(range(k, len(v["truth"]) - k))
        sets.append(CropDataset(v["images"], labels, frames, k=k, crop=crop, n_samples=iters * bs // len(train_vols) + bs,
                                seed=seed + i, stats=stats, scale_range=(0.8, 1.25)))
    dl = DataLoader(ConcatDataset(sets), batch_size=bs, shuffle=True, num_workers=0)
    model = build_model({**cfg, "in_channels": 2 * k + 1, "num_classes": 3}).to(dev)
    if init is not None:   # fine-tuning: start from another twin's weights
        model.load_state_dict(torch.load(init, map_location=dev, weights_only=False)["model"])
    w = torch.tensor([1.0, 1.0, 1.0], device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.1)
    best, t0, log = -1, time.time(), []
    model.train()
    for it, batch in enumerate(dl, 1):
        x, y = batch[0].to(dev), batch[1].to(dev)
        logits = model(x, batch[2].to(dev)) if stats else model(x)
        loss = dice_ce_loss(logits, y, w)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if it % 500 == 0 or it == iters:
            model.eval()
            ious = [score(predict(model, {"k": k}, v["images"], dev)[k:len(v["truth"]) - k],
                          v["truth"][k:len(v["truth"]) - k])["pore_iou"] for v in val_vols]
            model.train()
            log.append({"iter": it, "loss": float(loss.item()), "val_pore_iou": float(np.mean(ious))})
            print(f"  {name} it {it} loss {loss.item():.3f} val pore IoU {np.mean(ious):.4f} ({time.time()-t0:.0f}s)", flush=True)
            if np.mean(ious) > best:
                best = float(np.mean(ious))
                torch.save({"model": model.state_dict(), "config": model.config, "k": k, "iter": it, "crop": crop,
                            "trained_on": "digital twin (exact truth)"}, path)
        if it >= iters:
            break
    (OUT / name / "log.json").write_text(json.dumps(log, indent=1))
    return load_checkpoint(path, dev)


# ----------------------------------------------------------------------------- real-data box
def real_box(stack_frames=range(0, 130)):
    """Catalyst-layer interior of comp_full that every method covers (incl. pore_pipeline's ROI)."""
    stack = load_stack(ROOT / "data/raw/comp_full.tif", stack_frames)
    gdir = ROOT / "outputs/comp/masks/consensus"
    from eponge.io import load_masks as LM
    groi = LM(gdir)
    roi = np.ones(stack.shape[1:], bool)
    for z in stack_frames:
        if z in groi:
            roi &= groi[z] != 128
    from eponge import metrics as M
    rs, cs = M.largest_inner_rectangle(roi)
    return stack, rs, cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--out", default="runs/twin")
    ap.add_argument("--seed", type=int, default=0, help="training seed (data and test set unchanged)")
    ap.add_argument("--geometry", choices=["grf", "real", "slicegan"], default="grf",
                    help="grf: random-field structures; real: structures cut from the segmented comp_full stack "
                         "(train slices 0-179, val 180-249, test 250-353) mixed with random-field ones for training")
    a = ap.parse_args()
    global OUT
    OUT = ROOT / a.out
    dev = _device(a.device)
    OUT.mkdir(parents=True, exist_ok=True)
    calib = json.loads((OUT / "calibration.json").read_text()) if (OUT / "calibration.json").exists() else None
    if calib is None:
        v = load_stack(ROOT / "data/raw/comp_full.tif", range(60, 80))
        m = {z: export_to_classes(x) for z, x in load_masks(ROOT / "data/annotations_v2").items() if 60 <= z < 80}
        roi = np.stack([m[z] != 2 for z in range(60, 80)])
        geo = None
        if a.geometry in ("real", "slicegan"):
            cls = np.load(ROOT / "runs/twin/comp_full_millnet_classes.npy")
            geo = SIM.real_geometry(cls, range(0, 180), 1, depth=56, width=160, seed=99)[0]
        p, calib = SIM.calibrate(v[roi][::7], solid=geo)
        (OUT / "calibration.json").write_text(json.dumps(calib, indent=1))
    p = SIM.Imaging(**calib["params"])
    print("calibration (real vs twin grey percentiles):", calib["target_percentiles"], calib["sim_percentiles"], flush=True)

    t0 = time.time()
    if a.geometry == "real":
        cls = np.load(ROOT / "runs/twin/comp_full_millnet_classes.npy")
        g_train = SIM.real_geometry(cls, range(0, 180), 10, depth=64, seed=1)
        g_val = SIM.real_geometry(cls, range(180, 250), 2, depth=64, seed=2)
        g_test = SIM.real_geometry(cls, range(250, 354), 4, depth=64, seed=3)
        train = SIM.make_dataset(0, seed=1, p=p, geometries=g_train) + SIM.make_dataset(4, seed=11, p=p)
        val = SIM.make_dataset(0, seed=2, p=p, geometries=g_val)
        test = SIM.make_dataset(0, seed=3, p=p, geometries=g_test)
    elif a.geometry == "slicegan":
        # train and validate on SliceGAN-generated IrO2 structures (independent of our segmentation),
        # test on held-out real-derived structures (slices 250-353)
        z = np.load(ROOT / "runs/slicegan/volumes.npz")
        gan = [z[k] for k in sorted(z.files, key=lambda s: int(s.split("_")[1]))]
        cls = np.load(ROOT / "runs/twin/comp_full_millnet_classes.npy")
        train = SIM.make_dataset(0, seed=1, p=p, geometries=gan[:-2]) + SIM.make_dataset(4, seed=11, p=p)
        val = SIM.make_dataset(0, seed=2, p=p, geometries=gan[-2:])
        test = SIM.make_dataset(0, seed=3, p=p, geometries=SIM.real_geometry(cls, range(250, 354), 4, depth=64, seed=3))
    else:
        train = SIM.make_dataset(10, seed=1, p=p)
        val = SIM.make_dataset(2, seed=2, p=p)
        test = SIM.make_dataset(4, seed=3, p=p)
    print(f"twin datasets: {len(train)}+{len(val)}+{len(test)} volumes; shapes {sorted({v['images'].shape for v in train + test})} "
          f"test porosity {np.mean([v['truth'].mean() for v in test]):.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- models trained on the twin's exact truth
    sfx = f"_s{a.seed}" if a.seed else ""
    tw = lambda n, cfg, k: train_on_twin(n + sfx, cfg, k, train, val, dev, a.iters, seed=a.seed)
    twin_models = {
        "ResUNet 2D (twin truth)": tw("resunet_k0", {"arch": "resunet", "base": 24, "depth": 5, "max_ch": 128}, 0),
        "MillNet 2D (twin truth)": tw("millnet_k0", {"arch": "millnet"}, 0),
        "MillNet 7-slice, no streak bank (twin truth)": tw("millnet_k3_nobank", {"arch": "millnet", "use_bank": False}, 3),
        "MillNet 7-slice (twin truth)": tw("millnet_k3", {"arch": "millnet"}, 3),
    }
    real_models = {
        "MillNet 2D (real v2 labels)": load_checkpoint(ROOT / "runs/millnet_k0/best.pt", dev),
        "MillNet 7-slice (real v2 labels)": load_checkpoint(ROOT / "runs/millnet_k3_nobank/best.pt", dev),
        "ResUNet 2D (real v2 labels)": load_checkpoint(ROOT / "runs/resunet_k0/best.pt", dev),
    }

    # ---- Part A: exact truth on held-out twin volumes (margin of 3 slices for 7-slice context)
    K = 3
    results_a = {}
    def run_a(name, fn):
        preds, truths = [], []
        for v in test:
            pr = fn(v["images"])
            preds.append(pr[K:len(pr) - K]); truths.append(v["truth"][K:len(pr) - K])
        # volumes can differ in height (real layer thickness varies): pool pixel counts, average shape statistics
        P = np.concatenate([x.ravel() for x in preds]); T = np.concatenate([x.ravel() for x in truths])
        per_vol = [score(pp, tt) for pp, tt in zip(preds, truths)]
        r = {"pore_iou": float((P & T).sum() / (P | T).sum()), "accuracy": float((P == T).mean()),
             "porosity_pred": float(P.mean()), "porosity_true": float(T.mean()),
             "porosity_bias": float(P.mean() - T.mean()),
             "porosity_mae": float(np.mean([x["porosity_mae"] for x in per_vol]))}
        r["pore_iou_sd_over_volumes"] = float(np.std([x["pore_iou"] for x in per_vol]))
        r["xz_anisotropy_true"] = float(np.mean([anisotropy(tt)["xz_anisotropy"] for tt in truths]))
        r["xz_anisotropy_pred"] = float(np.mean([x["xz_anisotropy_pred"] for x in per_vol]))
        results_a[name] = r
        print(f"A | {name:48s} pore IoU {r['pore_iou']:.3f}  porosity {r['porosity_pred']:.3f} (true {r['porosity_true']:.3f}, "
              f"bias {r['porosity_bias']:+.3f})  x-z anisotropy {r['xz_anisotropy_pred']:.3f} (true {r['xz_anisotropy_true']:.3f})", flush=True)

    run_a("Otsu (denoised)", otsu_denoised)
    fs = {s: None for s in (1.0, -1.0)}
    for s in fs:
        pr = [ferner_style(v["images"], s) for v in test[:1]]
        fs[s] = score(pr[0][K:-K], test[0]["truth"][K:len(pr[0]) - K])["pore_iou"]
    best_shift = max(fs, key=fs.get)
    print(f"Ferner-style streak sign: IoU {fs} -> using shift {best_shift}", flush=True)
    run_a("Ferner-style z-gradient (pore_pipeline)", lambda st: ferner_style(st, best_shift))
    for name, (m, ck) in {**real_models, **twin_models}.items():
        run_a(name, lambda st, m=m, ck=ck: predict(m, ck, st, dev))
    (OUT / f"part_a_exact_truth{sfx}.json").write_text(json.dumps(results_a, indent=1))

    # ---- Part B: real comp_full interior, label-free
    stack, rs, cs = real_box()
    box = stack[:, rs, cs]
    zs = slice(3, len(box) - 3)
    print(f"real box: frames 0-{len(box)-1}, rows {rs.start}-{rs.stop}, cols {cs.start}-{cs.stop}", flush=True)
    results_b = {}
    def run_b(name, pore):
        pore = pore[zs]
        an = anisotropy(pore)
        per = pore.mean((1, 2))
        results_b[name] = {"porosity": float(pore.mean()), **an, "porosity_sd_over_z": float(per.std())}
        print(f"B | {name:48s} porosity {pore.mean():.3f}  x-z anisotropy {an['xz_anisotropy']:.3f}  "
              f"corr len x {an['corr_len_x']:.2f} z {an['corr_len_z']:.2f}  porosity sd over z {per.std():.4f}", flush=True)
    run_b("Otsu (denoised)", otsu_denoised(box))
    gf = load_masks(ROOT / "outputs/comp/masks/ferner_style")
    run_b("Ferner-style z-gradient (pore_pipeline)", np.stack([gf[z][rs, cs] == 0 for z in range(len(box))]))
    for name, (m, ck) in real_models.items():      # trained on full frames: predict full frames, then crop
        run_b(name, predict(m, ck, stack, dev)[:, rs, cs])
    for name, (m, ck) in twin_models.items():      # trained on layer interiors: predict the interior box
        run_b(name, predict(m, ck, box, dev))
    results_b["_reference"] = {"ferner_2024_published_porosity": 0.417, "box": [rs.start, rs.stop, cs.start, cs.stop]}
    (OUT / f"part_b_real_label_free{sfx}.json").write_text(json.dumps(results_b, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
