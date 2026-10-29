"""Score the team's MillNet-SA2 (real-trained, 3 seeds) against exact synthetic truth on the
seven v7d packages, next to the threshold rule and our v7-trained U-Nets, in the team's shared
metric set (3-class mIoU, in-layer pore IoU, layer IoU, porosity error) plus recall by floor depth.
    PFIB_EPONGE_SRC=<team repo, branch eponge-v2, commit 7892e9e>/eponge/src \
    PFIB_MILLNET_CKPT=<millnet_sa2_final_checkpoints> \
    python experiments/score_millnet.py OUT.json [n_frames]
Packages are read from $PFIB_OUT/<material> (generate.py) or $PFIB_OUT/team_review_synth_<material>.
MillNet runs exactly as the team reports it: eponge.inference.segment_stack, 1024 px tiles, k = 3, flip TTA.
The U-Net checkpoints (pfibsynth/checkpoints/*_v7_s0.pt) are optional; rows for missing ones are skipped.
"""
import os
import sys, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import ndimage as ndi
G = Path(__file__).resolve().parent.parent / "pfibsynth"
sys.path.insert(0, str(G)); sys.path.insert(1, os.environ["PFIB_EPONGE_SRC"])
import eval_shine as ev, adabn_eval as ab
from shine_depth_eval import depth_map, BINS
from eponge.models import build_model
from eponge.inference import segment_stack
import tifffile

out, NF = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16
CK = Path(os.environ["PFIB_MILLNET_CKPT"])
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
mill = {}
for s in (0, 1, 2):
    ck = torch.load(CK / f"seed{s}/best.pt", map_location="cpu", weights_only=True)
    m = build_model(ck["config"]).to(dev).eval(); m.load_state_dict(ck["model"]); mill[f"MillNet-SA2 s{s}"] = (m, ck["k"])
ours = {k: ev.load(G / "checkpoints" / f"{k}.pt", dev) for k in ("2d_random_v7_s0", "zwin_random_v7_s0") if (G / "checkpoints" / f"{k}.pt").exists()}

def iou(a, b):
    u = (a | b).sum(); return float((a & b).sum() / u) if u else float("nan")

res = {}
for name in ab.PKGS:
    t0 = time.time()
    pkg = next(p for p in (ev.DL / name, ev.DL / f"team_review_synth_{name}") if p.exists())
    meta = json.loads((pkg / "stack.json").read_text()); tj = json.loads((pkg / "truth/truth.json").read_text())
    shift = float(tj["optics"]["drift"]) + float(tj["optics"]["parallax"])
    tif = pkg / meta["stack_file"]; zl = ev.package_frames(NF)
    truth = {z: np.array(Image.open(pkg / f"truth/masks/slice_{z:04d}.png")) for z in range(min(zl), min(max(zl) + 9, 118))}
    tl = [truth[z] for z in range(min(truth), max(truth) + 1)]; off = min(truth)
    depths = [depth_map(tl, z - off, shift) for z in zl]
    T = [truth[z] for z in zl]
    keep = [ndi.binary_erosion(t != 128, iterations=ev.MARGIN) for t in T]
    tcls = [np.where(t == 0, 1, np.where(t == 255, 0, 2)).astype(np.uint8) for t in T]
    # predictions: pore masks (all), and 3-class maps where the method has them
    pore, cls3 = {}, {}
    sm = [np.array(Image.open(pkg / f"start_masks/slice_{z:04d}.png")) for z in zl]
    pore["rule (start masks)"] = [s == 0 for s in sm]
    cls3["rule (start masks)"] = [np.where(s == 0, 1, np.where(s == 255, 0, 2)).astype(np.uint8) for s in sm]
    drift = ab.band_drift(tif, truth[zl[0]] != 128)
    X = ab.stack_windows(tif, zl, drift, [truth[z] != 128 for z in zl])
    for k, (m, ck) in ours.items():
        pore[k] = list(ev.predict(m, ck, X, dev) > 0.5)
    stack = tifffile.imread(tif)
    for k, (m, kk) in mill.items():
        c, _ = segment_stack(m, stack, k=kk, device=dev, tta=True, frames=zl)
        cls3[k] = list(c); pore[k] = [ci == 1 for ci in c]
    r = {}
    for meth, P in pore.items():
        row = {}
        for lab, lo, hi in BINS:
            sel = [(d >= lo) & (d <= hi) & kp for d, kp in zip(depths, keep)]
            n = sum(s.sum() for s in sel)
            row[f"recall_{lab}"] = float(sum((p & s).sum() for p, s in zip(P, sel)) / max(n, 1))
        sol = [(d == 0) & kp for d, kp in zip(depths, keep)]
        row["solid_correct"] = float(sum((~p & s).sum() for p, s in zip(P, sol)) / max(sum(s.sum() for s in sol), 1))
        tp = np.concatenate([(t == 0)[kp] for t, kp in zip(T, keep)]); pp = np.concatenate([p[kp] for p, kp in zip(P, keep)])
        row["pore_iou_in_layer"] = iou(tp, pp)
        rp = np.concatenate([p[kp] for p, kp in zip(pore["rule (start masks)"], keep)]); row["pore_iou_vs_rule"] = iou(rp, pp)
        row["porosity_true"] = float(tp.mean()); row["porosity_pred"] = float(pp.mean())
        row["porosity_error_pp"] = 100 * (row["porosity_pred"] - row["porosity_true"])
        if meth in cls3:
            C = np.concatenate([c.ravel() for c in cls3[meth]]); Tc = np.concatenate([t.ravel() for t in tcls])
            ious = [iou(C == j, Tc == j) for j in range(3)]
            row["miou_3class"] = float(np.mean(ious)); row["iou_solid_pore_outside"] = ious
            row["layer_iou"] = iou(C != 2, Tc != 2)
            # porosity inside the method's own layer vs truth inside the true layer
            row["porosity_own_layer_error_pp"] = 100 * (float((C == 1).sum() / max((C != 2).sum(), 1)) - float((Tc == 1).sum() / max((Tc != 2).sum(), 1)))
        r[meth] = row
    res[name] = r
    print(name, f"{time.time()-t0:.0f}s", flush=True)
    for meth, row in r.items():
        print(f"   {meth:<20} " + " ".join(f"d{b[0]}={row['recall_' + b[0]]:.2f}" for b in BINS)
              + f" solid={row['solid_correct']:.2f} poreIoU={row['pore_iou_in_layer']:.3f} err={row['porosity_error_pp']:+.1f}"
              + (f" mIoU={row['miou_3class']:.3f} layer={row['layer_iou']:.3f}" if "miou_3class" in row else ""), flush=True)
    Path(out).write_text(json.dumps(res, indent=2))
