"""External benchmark on the public hand-labelled FIB-SEM data of Röding et al. (2021) / Skärberg et al. (2021).

Data: EC/HPC porous polymer films (HPC22, HPC30, HPC45), 200 slices of 2247 x 3372 px (10 nm pixels,
50 nm slices), 100 hand-labelled 256 x 256 squares per sample, each drawn with +-3 slices of context.
Split: the official train/val/test squares (60/20/20 per sample) shipped with Skärberg et al.'s code.
Metric: class-symmetric mean IoU over all test squares (Röding et al., eq. 1).

Published references on the same test squares:
  Röding et al. 2021, random forest + postprocessing:   0.7616 (Table 3)
  Skärberg et al. 2021, CNN (scored here from their released segmentation): computed by this script

    python eponge/scripts/external_benchmark.py --data ../external/skarberg2021/data
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
from eponge.models import build_model, count_parameters  # noqa: E402

M = 64          # context margin around each labelled square (pixels)
SQ = 256
IGNORE = 255


def miou(m, a):
    """Class-symmetric mean IoU (1 = solid), as in Röding et al. 2021."""
    i1, u1 = (m & a).sum(), (m | a).sum()
    i0, u0 = (~m & ~a).sum(), (~m | ~a).sum()
    return 0.5 * (i1 / u1 + i0 / u0)


def load_regions(data, n, kmax=3):
    """Regions (2*kmax+1, 256+2M, 256+2M) around each square, with the authors' preprocessing
    (linear background gradient along x removed, intensities scaled to [0, 1] per volume)."""
    R = np.load(data / f"manual_segmentation/segmentation_regions_HPC{n}.npy").astype(int)
    lab = np.load(data / f"manual_segmentation/manual_segmentation_HPC{n}.npy").astype(np.uint8)   # 1 = solid
    nz = 200
    need = {}
    for s in range(R.shape[1]):
        for dz in range(-kmax, kmax + 1):
            need.setdefault(int(np.clip(R[2, s] + dz, 0, nz - 1)), []).append((s, dz))
    first = np.array(Image.open(data / f"raw/HPC{n}/HPC{n}_0001.tif"), np.float32)
    nx, ny = first.shape
    # background gradient along x (rows), estimated on a sample of slices as in preprocess_data.py
    prof = np.mean([np.array(Image.open(data / f"raw/HPC{n}/HPC{n}_{z:04d}.tif"), np.float32).mean(1)
                    for z in range(1, nz + 1, 20)], 0)
    beta = np.polyfit(np.arange(1, nx + 1), prof, 1)
    bg = (beta[1] + np.arange(1, nx + 1) * beta[0])[:, None]
    regions = np.zeros((R.shape[1], 2 * kmax + 1, SQ + 2 * M, SQ + 2 * M), np.float32)
    lo = hi = None
    for z, uses in sorted(need.items()):
        img = np.array(Image.open(data / f"raw/HPC{n}/HPC{n}_{z + 1:04d}.tif"), np.float32) - bg + bg.mean()
        img = np.pad(img, M, mode="reflect")
        if lo is None:
            lo, hi = np.percentile(img[::8, ::8], [0.5, 99.5])
        for s, dz in uses:
            x, y = R[0, s] + M, R[1, s] + M
            regions[s, dz + kmax] = img[x - M:x + SQ + M, y - M:y + SQ + M]
    regions = np.clip((regions - lo) / (hi - lo), 0, 1).astype(np.float32)
    split = {k: np.load(data / f"regions/ind_squares_{k}_HPC{n}.npy").ravel().astype(int) for k in ("train", "val", "test")}
    return regions, lab, split


def target(lab_sq):
    """3-class target over the padded region: 0 solid, 1 pore inside the square, IGNORE in the margin."""
    t = np.full((SQ + 2 * M, SQ + 2 * M), IGNORE, np.int64)
    t[M:M + SQ, M:M + SQ] = np.where(lab_sq == 1, 0, 1)
    return t


def loss_fn(logits, y):
    ce = F.cross_entropy(logits, y, ignore_index=IGNORE)
    p = logits.softmax(1)[:, 1]
    v = y != IGNORE
    t = (y == 1).float()
    dice = 1 - (2 * (p * t)[v].sum() + 1) / (p[v].sum() + t[v].sum() + 1)
    return ce + dice


def stats_of(x):
    c = x[:, x.shape[1] // 2]
    q = torch.linspace(0.02, 0.98, 16, device=x.device)
    return torch.quantile(c.flatten(1).float().cpu(), q.cpu(), dim=1).T.to(x.device)


@torch.no_grad()
def predict(model, X, k, kmax, dev, needs_stats):
    model.eval()
    out = []
    for i in range(0, len(X), 8):
        x = torch.from_numpy(X[i:i + 8, kmax - k:kmax + k + 1]).to(dev)
        args = (x, stats_of(x)) if needs_stats else (x,)
        p = model(*args).softmax(1) + model(*[a.flip(-1) if a.dim() == 4 else a for a in args]).softmax(1).flip(-1)
        out.append(p[:, :2].argmax(1)[:, M:M + SQ, M:M + SQ].cpu().numpy())
    return np.concatenate(out) == 0   # True = solid


def strong_aug(x, rng):
    """Intensity-only strong augmentation (geometry unchanged, so teacher labels stay aligned)."""
    b = x.shape[0]
    g = np.exp(rng.uniform(-0.5, 0.5, (b, 1, 1, 1))).astype(np.float32)
    c = rng.uniform(0.7, 1.3, (b, 1, 1, 1)).astype(np.float32)
    o = rng.uniform(-0.12, 0.12, (b, 1, 1, 1)).astype(np.float32)
    x = (np.clip(x, 0, 1) ** g - 0.5) * c + 0.5 + o
    x = x + rng.normal(0, rng.uniform(0.01, 0.06), x.shape).astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


def train_eval(name, cfg, k, data_sets, dev, iters, kmax, out, train_frac=1.0, seed=0, init=None, ssl=False,
               ssl_tau=0.9, ssl_lambda=1.0, ema=0.995):
    """``ssl``: mean-teacher self-training (FixMatch-style) on the training squares whose labels are
    withheld at this label fraction: the EMA teacher labels weakly augmented crops where confident (>= tau),
    the student must match under strong intensity augmentation."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xtr, Ytr, Xva, Mva, Xte, Mte, Xun = [], [], [], [], [], [], []
    for n, (reg, lab, split) in data_sets.items():
        tr_all = split["train"]
        tr = tr_all[: max(1, int(round(len(tr_all) * train_frac)))]
        Xun.append(reg[tr_all[len(tr):]])
        Xtr.append(reg[tr]); Ytr.append(np.stack([target(lab[s]) for s in tr]))
        Xva.append(reg[split["val"]]); Mva.append(lab[split["val"]].astype(bool))
        Xte.append(reg[split["test"]]); Mte.append(lab[split["test"]].astype(bool))
    Xtr, Ytr = np.concatenate(Xtr), np.concatenate(Ytr)
    Xun = np.concatenate([x for x in Xun if len(x)]) if any(len(x) for x in Xun) else None
    Xva, Mva, Xte, Mte = map(np.concatenate, (Xva, Mva, Xte, Mte))
    model = build_model({**cfg, "in_channels": 2 * k + 1, "num_classes": 3}).to(dev)
    if init:
        sd = torch.load(init, map_location=dev, weights_only=False)["model"]
        own = model.state_dict()
        model.load_state_dict({kk: v for kk, v in sd.items() if kk in own and own[kk].shape == v.shape}, strict=False)
    needs_stats = getattr(model, "needs_stats", False) and getattr(model, "use_stats", False)
    teacher = None
    if ssl and Xun is not None:
        import copy
        teacher = copy.deepcopy(model).eval()
        for prm in teacher.parameters():
            prm.requires_grad_(False)
    lr = 1e-3 if init else 2e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.1)
    best, best_state, t0 = -1, None, time.time()
    for it in range(1, iters + 1):
        model.train()
        idx = rng.integers(0, len(Xtr), 8)
        x = Xtr[idx, kmax - k:kmax + k + 1].copy(); y = Ytr[idx].copy()
        flip = rng.random(8) < 0.5
        x[flip] = x[flip][..., ::-1]; y[flip] = y[flip][..., ::-1]
        g = np.exp(rng.uniform(-0.3, 0.3, (8, 1, 1, 1))).astype(np.float32)
        x = np.clip(x, 0, 1) ** g
        xt = torch.from_numpy(np.ascontiguousarray(x)).to(dev); yt = torch.from_numpy(np.ascontiguousarray(y)).to(dev)
        logits = model(xt, stats_of(xt)) if needs_stats else model(xt)
        loss = loss_fn(logits, yt)
        if teacher is not None:
            ui = rng.integers(0, len(Xun), 8)
            xu = Xun[ui, kmax - k:kmax + k + 1]
            fl = rng.random(8) < 0.5
            xu = xu.copy(); xu[fl] = xu[fl][..., ::-1]
            xw = torch.from_numpy(np.ascontiguousarray(xu)).to(dev)
            xs = torch.from_numpy(np.ascontiguousarray(strong_aug(xu, rng))).to(dev)
            with torch.no_grad():
                pt = (teacher(xw, stats_of(xw)) if needs_stats else teacher(xw)).softmax(1)[:, :2]
                conf, pl = pt.max(1)
            ls = model(xs, stats_of(xs)) if needs_stats else model(xs)
            mask = (conf >= ssl_tau).float()
            ramp = min(1.0, it / (0.2 * iters))
            lu = (F.cross_entropy(ls, pl, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
            loss = loss + ssl_lambda * ramp * lu
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if teacher is not None:
            with torch.no_grad():
                for tp, sp in zip(teacher.state_dict().values(), model.state_dict().values()):
                    if tp.dtype.is_floating_point:
                        tp.mul_(ema).add_(sp.detach(), alpha=1 - ema)
                    else:
                        tp.copy_(sp)
        if it % 250 == 0 or it == iters:
            v = miou(Mva, predict(teacher if teacher is not None else model, Xva, k, kmax, dev, needs_stats))
            if v > best:
                src = teacher if teacher is not None else model
                best, best_state = v, {kk: vv.detach().clone() for kk, vv in src.state_dict().items()}
            print(f"  {name} it {it} loss {loss.item():.3f} val mIoU {v:.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    pred = predict(model, Xte, k, kmax, dev, needs_stats)
    res = {"test_mIoU": float(miou(Mte, pred)), "val_mIoU": float(best), "params_M": count_parameters(model) / 1e6,
           "porosity_test_manual": float(1 - Mte.mean()), "porosity_test_pred": float(1 - pred.mean()),
           "train_squares": int(len(Xtr)), "k": k}
    per = {}
    o = 0
    for n, (_, _, split) in data_sets.items():
        m = len(split["test"]); per[f"HPC{n}"] = float(miou(Mte[o:o + m], pred[o:o + m])); o += m
    res["per_sample"] = per
    print(f"TEST {name}: mIoU {res['test_mIoU']:.4f} per sample {per}", flush=True)
    torch.save({"model": model.state_dict(), "config": model.config, "k": k}, out / f"{name}.pt")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "external/skarberg2021/data"))
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--out", default=str(ROOT / "runs/external"))
    ap.add_argument("--only", default=None, help="comma-separated run names")
    ap.add_argument("--fractions", default="1.0", help="training-label fractions, e.g. 0.1,0.25,1.0")
    ap.add_argument("--ssl", action="store_true", help="mean-teacher self-training on the unlabelled training squares")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", choices=["none", "twin"], default="none",
                    help="twin: start from the IrO2 digital-twin-trained weights of the same architecture")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    kmax = 3
    t0 = time.time()
    data_sets = {n: load_regions(data, n, kmax) for n in (22, 30, 45)}
    print(f"loaded regions ({time.time()-t0:.0f}s)", flush=True)

    # published reference, scored on the same squares from the authors' released segmentation
    from glob import glob
    ref_m, ref_a = [], []
    for n, (_, lab, split) in data_sets.items():
        R = np.load(data / f"manual_segmentation/segmentation_regions_HPC{n}.npy").astype(int)
        for s in split["test"]:
            seg = np.array(Image.open(data / f"seg/HPC{n}/seg_HPC{n}_{R[2, s]:04d}.png"))
            ref_m.append(lab[s].astype(bool)); ref_a.append(seg[R[0, s]:R[0, s] + SQ, R[1, s]:R[1, s] + SQ] > 0)
    results = {"Röding 2021 random forest (published, Table 3)": {"test_mIoU": 0.7616},
               "Skärberg 2021 CNN (released segmentation)": {"test_mIoU": float(miou(np.stack(ref_m), np.stack(ref_a)))}}
    print("reference", results, flush=True)

    runs = {
        "resunet_2d": ({"arch": "resunet", "base": 24, "depth": 5, "max_ch": 128}, 0),
        "resunet_7slice": ({"arch": "resunet", "base": 24, "depth": 5, "max_ch": 128}, 3),
        "millnet_2d": ({"arch": "millnet", "use_bank": False}, 0),
        "millnet_7slice": ({"arch": "millnet", "use_bank": False}, 3),
    }
    only = a.only.split(",") if a.only else list(runs)
    for frac in [float(f) for f in a.fractions.split(",")]:
        for name in only:
            cfg, k = runs[name]
            tag = ((name if frac == 1.0 else f"{name}_frac{frac}") + ("_twininit" if a.init == "twin" else "")
                   + ("_ssl" if a.ssl else "") + (f"_s{a.seed}" if a.seed else ""))
            init = None
            if a.init == "twin":
                tw = {"resunet_2d": "resunet_k0", "millnet_2d": "millnet_k0", "millnet_7slice": "millnet_k3_nobank"}
                if name not in tw:
                    continue
                init = ROOT / "runs/twin_real" / tw[name] / "best.pt"
            results[tag] = train_eval(tag, cfg, k, data_sets, dev, a.iters, kmax, out, frac, seed=a.seed, init=init, ssl=a.ssl)
            (out / f"results{'_' + a.init if a.init != 'none' else ''}{'_ssl' if a.ssl else ''}{'_s' + str(a.seed) if a.seed else ''}{'_' + a.fractions.replace(',', '-') if a.fractions != '1.0' else ''}.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v["test_mIoU"] for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
