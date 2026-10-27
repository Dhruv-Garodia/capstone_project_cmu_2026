"""Does frame context explain MillNet's low exact-truth score on the twin?

MillNet conditions on whole-frame grey-level quantiles. Real frames include the region outside the catalyst
layer; twin volumes are crops entirely inside it, so their frame statistics are unlike anything seen in
training. Here each twin crop (7 consecutive planes) is pasted into real test frames of the same acquisition,
inside the layer, and only the pasted region is scored against the exact truth. A model that does not use
frame statistics (U-Net ResNet-34) is the control.

    python eponge/scripts/twin_context_check.py   # -> runs/final_models/report/twin_context_check.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import report_final as RF  # noqa: E402
import train_final as TF  # noqa: E402
from eponge.inference import _device, predict_probs  # noqa: E402
from eponge.metrics import largest_inner_rectangle  # noqa: E402
from eponge.models import load_checkpoint  # noqa: E402
from eponge.preprocess import normalize  # noqa: E402

MODELS = ("millnet_sa2_long_k3_s0", "millnet_long_k3_s0", "unet_r34_long_k3_s0")


def main():
    dev = _device(None)
    twins = RF.twin_volumes(str(TF.ROOT / "runs/twin_refiner_v2"), "per_stack")
    data = {n: TF.load_stack(n) for n in TF.STACKS}
    out = {}
    for name in MODELS:
        model, ck = load_checkpoint(RF.RUNS / name / "best.pt", dev)
        k = ck.get("k", 3)
        res = {}
        for kind in ("images", "images_gan"):
            for mode in ("crop_alone", "pasted_in_real_frame"):
                per = {s: [] for s in TF.STACKS}
                for v in twins:
                    st = v["stack"]
                    imgs, masks, sp = data[st]
                    z0 = sp["test"][0]
                    rs, cs = largest_inner_rectangle(masks[z0] != 2)
                    h, w = min(rs.stop - rs.start, v[kind].shape[1]), min(cs.stop - cs.start, v[kind].shape[2])
                    y0, x0 = rs.start, cs.start + (cs.stop - cs.start - w) // 2
                    tw, tr = v[kind][:, :h, :w], v["truth"][:, :h, :w]
                    pred = []
                    for zz in range(k, len(tw) - k):
                        if mode == "crop_alone":
                            ch = np.stack([normalize(tw[q]) for q in range(zz - k, zz + k + 1)]).astype(np.float32)
                            pr = predict_probs(model, ch, dev, tta=False)
                        else:
                            frames = []
                            for j, q in enumerate(range(zz - k, zz + k + 1)):
                                f = imgs[min(z0 + j, len(imgs) - 1)].copy()
                                f[y0:y0 + h, x0:x0 + w] = tw[q]
                                frames.append(normalize(f))
                            pr = predict_probs(model, np.stack(frames).astype(np.float32), dev, tta=False)[:, y0:y0 + h, x0:x0 + w]
                        pred.append(pr[1] > pr[0])
                    p, t = np.stack(pred), tr[k:len(tw) - k].astype(bool)
                    per[st].append(float((p & t).sum() / max((p | t).sum(), 1)))
                res[f"{kind}/{mode}"] = {s: float(np.mean(x)) for s, x in per.items()}
                print(name, kind, mode, {s: round(x, 4) for s, x in res[f"{kind}/{mode}"].items()}, flush=True)
        out[name] = res
        (RF.OUT / "twin_context_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
