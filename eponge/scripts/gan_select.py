"""Use a fine-tuned GAN twin for one acquisition only if it is closer to real on validation slices.

Closeness = gap between a real-trained model's mIoU on GAN-twin images and on the real images of the same
held-out validation structures (twin_refiner.py downstream --split val). Writes <base>/per_stack.json.

    python eponge/scripts/gan_select.py --base runs/twin_refiner_v2 --cand runs/twin_refiner_v3_pristine --stack pristine
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "millnet_sa2_long_k3_s0"


def gap(run, stack):
    d = json.loads((ROOT / run / f"downstream_{MODEL}_val{'_noise1.0' if 'v2' in run else ''}.json").read_text())[stack]
    return d["gan_twin"]["mean_iou"] - d["real"]["mean_iou"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--cand", required=True); ap.add_argument("--stack", required=True)
    a = ap.parse_args()
    g0, g1 = gap(a.base, a.stack), gap(a.cand, a.stack)
    f = ROOT / a.base / "per_stack.json"
    m = json.loads(f.read_text()) if f.exists() else {}
    if abs(g1) < abs(g0):
        m[a.stack] = a.cand
    else:
        m.pop(a.stack, None)
    f.write_text(json.dumps(m, indent=1))
    print(f"{a.stack}: validation gap base {g0:+.4f}, candidate {g1:+.4f} -> using {'candidate' if a.stack in m else 'base'}")
