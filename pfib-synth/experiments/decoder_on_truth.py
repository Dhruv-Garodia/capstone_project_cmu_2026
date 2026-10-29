"""Does a decoder calibrated to agree with labels also get closer to true porosity?

The team froze MillNet-SA2 (3 seeds) with its argmax decoder and reports Yunzhi's shared
validation-selected decoder as an optional ablation (it lowered label-based porosity MAE on the
real test split, 1.614 -> 1.492 pp, and worsened pristine). This repeats that selection on our
synthetic stacks, where exact truth exists, in the team's four locked metrics (3-class mIoU,
pore IoU in the true layer, layer IoU, porosity MAE in pp; mean +- std over 3 seeds).

Decoder (Yunzhi, EXPERIMENTS.md 3.3): r = P(outside), q = P(pore | inside) = p1 / (p0 + p1);
inside = (1 - r) > layer_t, pore = inside and q > pore_t; pore_t 0.35-0.65 step 0.01, layer_t
0.3-0.7 step 0.1, plus argmax. One shared setting per seed, chosen on validation frames by mean
porosity MAE against the labels, keeping mIoU and pore IoU within 0.002 of argmax on every
material. Here the labels are the threshold-rule start masks (how our real labels began); we then
score every setting against truth on separate test frames. Her actual final-SA2 thresholds were
not in the shared files, and the reviewed real labels are not used, so this reproduces her
procedure, not her numbers.

    PFIB_EPONGE_SRC=... PFIB_MILLNET_CKPT=... PFIB_OUT=... python experiments/decoder_on_truth.py OUT.json
"""
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, tifffile
from PIL import Image
sys.path.insert(0, os.environ["PFIB_EPONGE_SRC"])
from eponge.models import build_model
from eponge.inference import predict_probs
from eponge.data import context_stack
from eponge.preprocess import normalize

CK = Path(os.environ["PFIB_MILLNET_CKPT"]); OUT = Path(os.environ.get("PFIB_OUT", "out"))
PKGS = ["comp_like", "uncomp_like", "pristine_like", "mudcrack", "fibre_mat", "fibre_coated", "faceted"]
TEST = list(np.linspace(3, 112, 16).astype(int))                       # the frames score_millnet.py uses
VAL = [6, 20, 35, 49, 64, 78, 93, 107]                                  # between them, 3+ slices from any test frame
PT = np.round(np.arange(0.35, 0.655, 0.01), 2); LT = np.round(np.arange(0.3, 0.75, 0.1), 1)
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def cls(m):  # team encoding: 0 pore, 255 solid, 128 outside -> 0 solid, 1 pore, 2 outside
    return np.where(m == 0, 1, np.where(m == 255, 0, 2)).astype(np.int64)


def counts(p, ref):
    """For every decoder setting, the 3x3 confusion (pred x ref) of one frame. Returns
    argmax (3,3) and grid (len(LT), len(PT), 3, 3) using a 0.001 histogram of q."""
    r, q = p[2], p[1] / np.maximum(p[0] + p[1], 1e-6)
    am = np.bincount((p.argmax(0) * 3 + ref).ravel(), minlength=9).reshape(3, 3)
    qb = np.clip((q * 1000).astype(np.int64), 0, 1000)
    g = np.zeros((len(LT), len(PT), 3, 3), np.int64)
    for i, lt in enumerate(LT):
        ins = (1 - r) > lt
        out = np.bincount(ref[~ins], minlength=3)
        h = np.bincount((ref[ins] * 1001 + qb[ins]), minlength=3003).reshape(3, 1001)   # ref class x q bin
        for j, pt in enumerate(PT):
            k = int(round(pt * 1000))                                      # q > pt, to within 0.001
            g[i, j, 1] = h[:, k:].sum(1); g[i, j, 0] = h[:, :k].sum(1); g[i, j, 2] = out
    return am, g


def metrics(C, poros):
    """C: (..., 3, 3) pooled confusion pred x ref; poros: (..., F, 2) per-frame [pred, ref] porosity."""
    tp = np.diagonal(C, axis1=-2, axis2=-1)
    iou = tp / np.maximum(C.sum(-1) + C.sum(-2) - tp, 1)
    ins_p, ins_r = C[..., :2, :].sum((-1, -2)), C[..., :, :2].sum((-1, -2))
    layer = C[..., :2, :2].sum((-1, -2)) / np.maximum(ins_p + ins_r - C[..., :2, :2].sum((-1, -2)), 1)
    tl = C[..., 1, 1] / np.maximum(C[..., 1, :2].sum(-1) + C[..., 0, 1] + C[..., 2, 1], 1)   # pore IoU in the true layer
    mae = 100 * np.abs(poros[..., 0] - poros[..., 1]).mean(-1)
    bias = 100 * (poros[..., 0] - poros[..., 1]).mean(-1)
    return dict(miou=iou.mean(-1), pore_iou=iou[..., 1], pore_iou_true_layer=tl, layer_iou=layer, mae=mae, bias=bias)


def frame_poros(cf):  # porosity in the predicted layer vs porosity in the reference layer, from one frame's confusion
    return np.stack([cf[..., 1, :].sum(-1) / np.maximum(cf[..., :2, :].sum((-1, -2)), 1),
                     cf[..., :, 1].sum(-1) / np.maximum(cf[..., :, :2].sum((-1, -2)), 1)], -1)


import pickle
cache = Path(sys.argv[1]).with_suffix(".counts.pkl")                    # per-frame confusion counts, so the selection can be re-run
raw = pickle.loads(cache.read_bytes()) if cache.exists() else {}
for s in (0, 1, 2) if not raw else ():
    ck = torch.load(CK / f"seed{s}/best.pt", map_location="cpu", weights_only=True)
    m = build_model(ck["config"]).to(dev).eval(); m.load_state_dict(ck["model"]); k = ck["k"]
    for name in PKGS:
        t0 = time.time()
        pkg = next(p for p in (OUT / name, OUT / f"team_review_synth_{name}") if p.exists())
        meta = json.loads((pkg / "stack.json").read_text())
        norm = np.stack([normalize(f) for f in tifffile.imread(pkg / meta["stack_file"])]).astype(np.float32)
        rec = {}
        for split, zl in (("val", VAL), ("test", TEST)):
            for ref in ("truth", "label"):
                rec[split, ref] = ([], [])
            for z in zl:
                p = predict_probs(m, context_stack(norm, z, k), dev, tta=True)
                for ref, d in (("truth", "truth/masks"), ("label", "start_masks")):
                    am, g = counts(p, cls(np.array(Image.open(pkg / d / f"slice_{z:04d}.png"))))
                    rec[split, ref][0].append(am); rec[split, ref][1].append(g)
        raw[s, name] = {key: (np.stack(a), np.stack(g)) for key, (a, g) in rec.items()}
        print(f"seed {s} {name} {time.time() - t0:.0f}s", flush=True)
if not cache.exists():
    cache.write_bytes(pickle.dumps(raw))

# selection per seed: one shared setting, label-based, on val frames, with the per-material guard
res = {"val_frames": [int(z) for z in VAL], "test_frames": [int(z) for z in TEST], "seeds": {}}
for s in (0, 1, 2):
    ok = np.ones((len(LT), len(PT)), bool); maes = []
    for name in PKGS:
        a, g = raw[s, name]["val", "label"]
        ma = metrics(a.sum(0), frame_poros(a)); mg = metrics(g.sum(0), np.moveaxis(frame_poros(g), 0, -2))
        ok &= (mg["miou"] >= ma["miou"] - 0.002) & (mg["pore_iou"] >= ma["pore_iou"] - 0.002)
        maes.append((ma["mae"], mg["mae"]))
    am_mae = np.mean([x[0] for x in maes]); g_mae = np.mean([x[1] for x in maes], 0)
    cand = np.where(ok, g_mae, np.inf)
    i, j = np.unravel_index(np.argmin(cand), cand.shape)
    sel = "argmax" if not np.isfinite(cand[i, j]) or cand[i, j] >= am_mae else (float(LT[i]), float(PT[j]))
    # truth-optimal shared setting on val (an oracle: what the labels cannot tell you)
    t_mae = np.mean([metrics(raw[s, n]["val", "truth"][1].sum(0), np.moveaxis(frame_poros(raw[s, n]["val", "truth"][1]), 0, -2))["mae"] for n in PKGS], 0)
    ti, tj = np.unravel_index(np.argmin(t_mae), t_mae.shape)
    out = {"label_selected": sel, "truth_oracle": (float(LT[ti]), float(PT[tj])), "val_label_mae": {"argmax": float(am_mae)},
           "materials": {}}
    if sel != "argmax":
        out["val_label_mae"]["selected"] = float(g_mae[i, j])
    for name in PKGS:
        row = {}
        for ref in ("truth", "label"):
            a, g = raw[s, name]["test", ref]
            pick = {"argmax": metrics(a.sum(0), frame_poros(a))}
            for lab, ij in (("calibrated", None if sel == "argmax" else (i, j)), ("truth_oracle", (ti, tj))):
                if ij is None:
                    pick[lab] = pick["argmax"]; continue
                gg = g[:, ij[0], ij[1]]; pick[lab] = metrics(gg.sum(0), frame_poros(gg))
            row[ref] = {d: {kk: float(v) for kk, v in mm.items()} for d, mm in pick.items()}
        out["materials"][name] = row
    res["seeds"][s] = out
    print(f"seed {s}: label-selected {sel}  truth-oracle {out['truth_oracle']}", flush=True)
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
print("wrote", sys.argv[1])
