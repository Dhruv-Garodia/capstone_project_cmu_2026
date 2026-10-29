"""Control: MillNet-SA2 vs the threshold rule (start masks) on REAL test frames (101-117), pore IoU inside the layer.
    PFIB_EPONGE_SRC=... PFIB_MILLNET_CKPT=... PFIB_REAL_DIR=... PFIB_LABELS_DIR=... python experiments/real_control.py OUT.json
"""
import os, sys, json
from pathlib import Path
import numpy as np, torch, tifffile
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, os.environ["PFIB_EPONGE_SRC"])
from eponge.models import build_model
from eponge.inference import segment_stack
CK = Path(os.environ["PFIB_MILLNET_CKPT"])
REAL, LABELS = Path(os.environ["PFIB_REAL_DIR"]), Path(os.environ["PFIB_LABELS_DIR"])
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
res = {}
for s in (0, 1, 2):
    ck = torch.load(CK / f"seed{s}/best.pt", map_location="cpu", weights_only=True)
    m = build_model(ck["config"]).to(dev).eval(); m.load_state_dict(ck["model"])
    for st, pk in (("comp", "team_review"), ("uncomp", "team_review_uncomp"), ("pristine", "team_review_pristine")):
        d = LABELS / pk
        test = json.loads((CK / f"splits/{st}_split.json").read_text())["test"]
        zl = [z for z in test if (d / f"start_masks/slice_{z:04d}.png").exists()]
        stack = tifffile.imread(REAL / f"{st}_full.tif", key=range(0, 118))
        c, _ = segment_stack(m, stack, k=ck["k"], device=dev, tta=True, frames=zl)
        R = [np.array(Image.open(d / f"start_masks/slice_{z:04d}.png")) for z in zl]
        keep = [ndi.binary_erosion(r != 128, iterations=10) for r in R]
        a = np.concatenate([(r == 0)[k] for r, k in zip(R, keep)]); b = np.concatenate([(ci == 1)[k] for ci, k in zip(c, keep)])
        iou = float((a & b).sum() / (a | b).sum())
        res.setdefault(st, []).append(iou); print(st, s, len(zl), round(iou, 3), "rule por", round(a.mean(), 3), "mill por", round(b.mean(), 3), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
