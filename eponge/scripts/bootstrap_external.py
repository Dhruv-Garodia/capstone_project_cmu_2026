"""Bootstrap confidence intervals for the public benchmark (60 test squares).

For every saved model in runs/external/ and for Skärberg et al.'s released segmentation:
95 % CI of the combined test mIoU by resampling test squares (2000 resamples), and a paired bootstrap of
the difference to Skärberg et al. (same resampled squares for both), so that "better" means the
difference's CI excludes zero.

    python eponge/scripts/bootstrap_external.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import external_benchmark as EB  # noqa: E402
from eponge.models import load_checkpoint  # noqa: E402


def counts(m, a):
    """Per-square intersection/union counts for both classes (1 = solid)."""
    return np.stack([(m & a).sum((1, 2)), (m | a).sum((1, 2)), (~m & ~a).sum((1, 2)), (~m | ~a).sum((1, 2))], 1)


def miou_from(c):
    s = c.sum(0)
    return 0.5 * (s[0] / s[1] + s[2] / s[3])


def main(n_boot=2000):
    data = ROOT.parent / "external/skarberg2021/data"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sets = {n: EB.load_regions(data, n, 3) for n in (22, 30, 45)}
    X, Mt, ref = [], [], []
    for n, (reg, lab, split) in sets.items():
        R = np.load(data / f"manual_segmentation/segmentation_regions_HPC{n}.npy").astype(int)
        for s in split["test"]:
            X.append(reg[s]); Mt.append(lab[s].astype(bool))
            seg = np.array(Image.open(data / f"seg/HPC{n}/seg_HPC{n}_{R[2, s]:04d}.png"))
            ref.append(seg[R[0, s]:R[0, s] + EB.SQ, R[1, s]:R[1, s] + EB.SQ] > 0)
    X, Mt, ref = np.stack(X), np.stack(Mt), np.stack(ref)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(X), (n_boot, len(X)))
    c_ref = counts(Mt, ref)
    boot_ref = np.array([miou_from(c_ref[i]) for i in idx])
    out = {"Skärberg 2021 CNN": {"mIoU": float(miou_from(c_ref)), "ci95": np.percentile(boot_ref, [2.5, 97.5]).tolist()}}
    for ck in sorted((ROOT / "runs/external").glob("*.pt")):
        m, c = load_checkpoint(ck, dev)
        k = c.get("k", 0)
        ns = getattr(m, "needs_stats", False) and getattr(m, "use_stats", False)
        pred = EB.predict(m, X, k, 3, dev, ns)
        cc = counts(Mt, pred)
        boot = np.array([miou_from(cc[i]) for i in idx])
        diff = boot - boot_ref
        out[ck.stem] = {"mIoU": float(miou_from(cc)), "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                        "diff_vs_skarberg": float(miou_from(cc) - miou_from(c_ref)),
                        "diff_ci95": np.percentile(diff, [2.5, 97.5]).tolist(),
                        "p_better_than_skarberg": float((diff > 0).mean())}
        r = out[ck.stem]
        print(f"{ck.stem:38s} mIoU {r['mIoU']:.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]  "
              f"vs Skärberg {r['diff_vs_skarberg']:+.4f} [{r['diff_ci95'][0]:+.4f}, {r['diff_ci95'][1]:+.4f}]", flush=True)
    print("Skärberg", out["Skärberg 2021 CNN"])
    (ROOT / "runs/external/bootstrap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
