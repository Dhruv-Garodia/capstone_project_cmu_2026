"""Segment all of comp_full with the 2D MillNet; source of real-derived geometry for the digital twin.

    python eponge/scripts/segment_full_stack.py --device cpu     # writes runs/twin/comp_full_millnet_classes.npy
"""
import argparse
from pathlib import Path

import numpy as np

from eponge.inference import segment_stack
from eponge.io import load_stack
from eponge.models import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); a = ap.parse_args()
model, ck = load_checkpoint(ROOT / "runs/millnet_k0/best.pt", a.device)
stack = load_stack(ROOT / "data/raw/comp_full.tif")
cls, _ = segment_stack(model, stack, 0, a.device, tta=False, progress=True)
(ROOT / "runs/twin").mkdir(parents=True, exist_ok=True)
np.save(ROOT / "runs/twin/comp_full_millnet_classes.npy", cls)
print("saved", cls.shape, "porosity", float((cls[cls != 2] == 1).mean()))
