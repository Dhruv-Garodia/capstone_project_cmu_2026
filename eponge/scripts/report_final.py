"""One evaluation pass over every model trained on the final dataset -> tables for the paper.

For each run in runs/final_models (and nnU-Net, from its saved predictions):
  1. test metrics per stack: mIoU, pore IoU, layer IoU, porosity error (same evaluate_frames for all);
  2. 95 % bootstrap CI of mIoU over test slices, and a paired bootstrap of the difference to the reference
     model (--ref, default nnU-Net 2D), so "better" means the CI of the difference excludes zero;
  3. physics on the 3D test block (slices 101-117): porosity, median pore diameter, specific surface,
     pore connectivity, tortuosity, each compared with the same quantity computed on the labels;
  4. digital twin with exact truth: SliceGAN volumes (held out from twin pretraining) and real-derived
     geometry rendered through each stack's calibrated imaging; pore IoU of every model against the
     exact phase, next to Otsu and the Ferner et al. rule;
  5. size and speed: parameters and seconds per full-frame slice.

    python eponge/scripts/report_final.py     # -> runs/final_models/report/{results.json, results.md, preds/}
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

sys.path.insert(0, str(Path(__file__).parent))
import train_final as TF  # noqa: E402
from eponge import ferner as FE  # noqa: E402
from eponge import simulate as SIM  # noqa: E402
from eponge.inference import _device, predict_probs, segment_stack  # noqa: E402
from eponge.metrics import analyze  # noqa: E402
from eponge.models import count_parameters, load_checkpoint  # noqa: E402
from eponge.preprocess import normalize  # noqa: E402

RUNS = TF.ROOT.parent / "runs/final_models"
OUT = RUNS / "report"
NB = 2000


def confusion(p, g):
    """Per-slice (3 x 3) confusion counts."""
    return np.bincount(3 * g.ravel().astype(np.int64) + p.ravel(), minlength=9).reshape(3, 3)


def miou_from(cm):
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    return float(np.mean(inter / np.maximum(union, 1)))


def pore_iou_in_roi(cm):
    """Pore IoU over pixels inside the true layer (solid/pore rows), as evaluate_frames reports it."""
    tp = cm[1, 1]; fp = cm[0, 1]; fn = cm[1, 0] + cm[1, 2]
    return float(tp / max(tp + fp + fn, 1))


def physics(cls):
    a = analyze(cls, pixel_nm=6.0, compute_tortuosity=True)
    t = a.get("tortuosity", {}).get("through_plane_y", {})
    return {"porosity": a["porosity"], "d50_nm": a["pore_size_nm"].get("d50"),
            "specific_surface_per_um": a["specific_surface_per_um"],
            "connected_pore_fraction": a["connectivity"].get("largest_fraction"),
            "tortuosity_through_plane": t.get("tau") if isinstance(t, dict) else None}


def twin_volumes(gan_run=None, geometry="comp_full"):
    z = np.load(TF.ROOT / "runs/slicegan/volumes.npz")
    gan = [z[k] for k in sorted(z.files, key=lambda s: int(s.split("_")[1]))][12:]          # held out
    cls = np.load(TF.ROOT / "runs/twin/comp_full_millnet_classes.npy")
    real = SIM.real_geometry(cls, range(250, 354), 2, depth=48, seed=77)
    vols, geos = [], gan + real
    G = None
    if gan_run:
        import twin_refiner as TR
        dev = _device(None)
        G = TR.load_G(dev, gan_run)
        rng = np.random.default_rng(9)
    for stack in TF.STACKS:
        if geometry == "per_stack":                       # held-out volumes of the acquisition's own SliceGAN
            zz = np.load(TF.ROOT / f"runs/slicegan_{stack}/volumes.npz")
            geos = [zz[k] for k in sorted(zz.files, key=lambda s: int(s.split("_")[1]))][12:14]
            gan, real = geos, []
        c = json.loads((TF.ROOT / f"runs/twin_{stack}/calibration.json").read_text())["params"]
        p = SIM.Imaging(**{k: v for k, v in c.items() if k in SIM.Imaging.__dataclass_fields__})
        for i, d in enumerate(SIM.make_dataset(0, seed=300, p=p, geometries=geos)):
            v = {"stack": stack, "geometry": "slicegan" if i < len(gan) else "real", **d}
            if G is not None:
                v["images_gan"] = TR.gan_volume(G, geos[i], stack, dev, rng)
            vols.append(v)
    return vols


def twin_pore_iou(pred_pore, truth, k):
    p, t = pred_pore[k:len(truth) - k], truth[k:len(truth) - k].astype(bool)
    return float((p & t).sum() / max((p | t).sum(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="nnunet_2d")
    ap.add_argument("--smoke", action="store_true", help="pipeline test: 2 runs, 2 test slices, 1 twin volume")
    ap.add_argument("--out", default=None)
    ap.add_argument("--twin-geometry", default="comp_full", choices=["comp_full", "per_stack"])
    ap.add_argument("--twin-gan", default=None, help="GAN-twin run folder: also score on GAN-twin images")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    dev = _device(None)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "preds").mkdir(exist_ok=True)
    data = {n: TF.load_stack(n) for n in TF.STACKS}
    tests = {n: [z for z in data[n][2]["test"] if z in data[n][1]][: (2 if a.smoke else None)] for n in TF.STACKS}
    rng = np.random.default_rng(0)
    boot_idx = {n: rng.integers(0, len(tests[n]), (NB, len(tests[n]))) for n in TF.STACKS}

    print("twin volumes...", flush=True)
    twins = twin_volumes(a.twin_gan, a.twin_geometry)[: (1 if a.smoke else None)]
    kinds = ["images"] + (["images_gan"] if a.twin_gan else [])
    fbest = json.loads((TF.ROOT / "runs/twin_real/ferner_oracle.json").read_text())["parameters"]
    res = {"models": {}, "labels_physics": {}, "twin_baselines": {}}
    for n in TF.STACKS:
        res["labels_physics"][n] = physics(np.stack([data[n][1][z] for z in tests[n]]))
    tb = {}
    for kind in kinds:
        sfx = "" if kind == "images" else "_gan"
        tb[f"otsu{sfx}"], tb[f"ferner_rule{sfx}"] = [], []
        for v in twins:
            sm = np.stack([ndi.gaussian_filter(f.astype(np.float32), 1) for f in v[kind]])
            tb[f"otsu{sfx}"].append(twin_pore_iou(sm < threshold_otsu(sm), v["truth"], 3))
            tb[f"ferner_rule{sfx}"].append(twin_pore_iou(FE.segment(v[kind], streak=v["imaging"]["streak"], **fbest), v["truth"], 3))
    res["twin_baselines"] = {k: float(np.mean(x)) for k, x in tb.items()}

    runs = sorted(r for r in RUNS.iterdir() if (r / "summary.json").exists() and not r.name.startswith("_")
                  and r.name != "report")
    if a.smoke:
        runs = runs[:2]
    for run in runs:
        summ = json.loads((run / "summary.json").read_text())
        test_stacks = summ["args"].get("test_stacks", "comp,uncomp,pristine").split(",")
        entry = {"train_stacks": summ["args"].get("train_stacks"), "test": {}}
        preds_all = {}
        if (run / "best.pt").exists():
            model, ck = load_checkpoint(run / "best.pt", dev); k = ck.get("k", 0)
            entry["params_M"] = count_parameters(model) / 1e6
            for n in test_stacks:
                t0 = time.time()
                cls, _ = segment_stack(model, data[n][0], k, dev, tta=True, frames=tests[n])
                entry.setdefault("sec_per_slice", (time.time() - t0) / len(tests[n]))
                preds_all[n] = cls
            # exact truth on the twin (pore vs solid decision; twin volumes contain no "outside");
            # main models only (seed 0, all stacks) to bound the run time
            do_twin = run.name.endswith("_s0") and not run.name.startswith(("loso_", "abl_"))
            for kind in (kinds if do_twin else []):
                sfx = "" if kind == "images" else "_gan"
                ious = []
                for v in twins:
                    norm = np.stack([normalize(f) for f in v[kind]]).astype(np.float32)
                    pp = []
                    for zz in range(len(norm)):
                        idx = np.clip(np.arange(zz - k, zz + k + 1), 0, len(norm) - 1)
                        pr = predict_probs(model, norm[idx], dev, tta=False)
                        pp.append(pr[1] > pr[0])
                    ious.append(twin_pore_iou(np.stack(pp), v["truth"], 3))
                entry[f"twin{sfx}_pore_iou"] = float(np.mean(ious))
                entry[f"twin{sfx}_pore_iou_by_stack"] = {s: float(np.mean([i for i, v in zip(ious, twins) if v["stack"] == s])) for s in TF.STACKS}
        elif (run / "preds.npz").exists():                                  # nnU-Net
            z = np.load(run / "preds.npz"); preds_all = {n: z[n] for n in test_stacks if n in z.files}
        for n, cls in preds_all.items():
            np.save(OUT / "preds" / f"{run.name}_{n}.npy", cls.astype(np.uint8))
            gts = [data[n][1][zz] for zz in tests[n]]
            cms = np.stack([confusion(p, g) for p, g in zip(cls, gts)])
            tot = cms.sum(0)
            boot = np.array([miou_from(cms[i].sum(0)) for i in boot_idx[n]])
            pr = [float((p[g != 2] == 1).mean()) for p, g in zip(cls, gts)]
            gp = [float((g[g != 2] == 1).mean()) for g in gts]
            entry["test"][n] = {"mean_iou": miou_from(tot), "mean_iou_ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                                "pore_iou": pore_iou_in_roi(tot),
                                "layer_iou": float(((tot[:2, :2].sum()) / max(tot[:2, :2].sum() + tot[:2, 2].sum() + tot[2, :2].sum(), 1))),
                                "porosity_error": float(np.mean(np.abs(np.array(pr) - np.array(gp)))),
                                "physics": physics(cls), "_cms": cms.tolist()}
        res["models"][run.name] = entry
        print(run.name, {n: round(v["mean_iou"], 4) for n, v in entry["test"].items()}, "twin", entry.get("twin_pore_iou"), flush=True)

    # paired bootstrap vs the reference model
    ref = res["models"].get(a.ref)
    for name, e in res["models"].items():
        for n, t in e["test"].items():
            if ref is None or n not in ref["test"] or name == a.ref:
                continue
            c1, c0 = np.array(t["_cms"]), np.array(ref["test"][n]["_cms"])
            d = np.array([miou_from(c1[i].sum(0)) - miou_from(c0[i].sum(0)) for i in boot_idx[n]])
            t[f"diff_vs_{a.ref}"] = {"mean": float(t["mean_iou"] - ref["test"][n]["mean_iou"]),
                                     "ci95": np.percentile(d, [2.5, 97.5]).tolist(), "p_better": float((d > 0).mean())}
    for e in res["models"].values():
        for t in e["test"].values():
            t.pop("_cms", None)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))

    # markdown table (all-stack models)
    lines = ["| model | params (M) | s/slice | comp mIoU | uncomp mIoU | pristine mIoU | mean | physics-twin pore IoU | GAN-twin pore IoU |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, e in sorted(res["models"].items(), key=lambda kv: -np.mean([t["mean_iou"] for t in kv[1]["test"].values()] or [0])):
        if e.get("train_stacks", "comp,uncomp,pristine") != "comp,uncomp,pristine" or len(e["test"]) < 3:
            continue
        m = [e["test"][n]["mean_iou"] for n in TF.STACKS]
        lines.append(f"| {name} | {e.get('params_M', float('nan')):.1f} | {e.get('sec_per_slice', float('nan')):.2f} | "
                     + " | ".join(f"{e['test'][n]['mean_iou']:.4f}" for n in TF.STACKS)
                     + f" | {np.mean(m):.4f} | {e.get('twin_pore_iou', float('nan')):.4f} | {e.get('twin_gan_pore_iou', float('nan')):.4f} |")
    lines += ["", "Twin baselines (exact truth): " + ", ".join(f"{k} {v:.4f}" for k, v in res["twin_baselines"].items())]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
