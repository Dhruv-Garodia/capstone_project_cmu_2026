"""Compare every trained model and baseline on the held-out test frames 100-117.

References (all on the same frames, no registration):
  v2        corrected labels (data/annotations_v2): solid / pore / outside, the training target
  v1        original napari labels (data/manual_annotation)
  garodia   D. Garodia's trimap (pore_pipeline: Otsu + Ferner-style consensus inside the hand
            border); scored on its *confident* pixels only (uncertain and excluded ignored).
            Produce it with:  python -m pore_pipeline.run data/raw/comp_full.tif --out outputs/comp --z1 130
                              python -m pore_pipeline.apply_borders data/raw/comp_full.tif outputs/comp \
                                  data/manual_annotation --out outputs/comp/manual_borders

Optional: --previous-ckpt points at the earlier real-data experiment's checkpoint (not in git)
to include it in the table.

Writes runs/comparison.json, docs/model_comparison.md (table) and webapp/model/results.json.

    python eponge/scripts/compare_models.py [--device mps]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from eponge.baselines import fixed_threshold, local_threshold, otsu_raw
from eponge.evaluate import evaluate_frames
from eponge.inference import segment_stack
from eponge.io import export_to_classes, load_masks, load_stack
from eponge.models import count_parameters, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
TEST = list(range(100, 118))
NAMES = {
    "resunet_k0": "ResUNet (Éponge, 2D)",
    "unet2d_v2": "ResUNet (Éponge, 2D, batch 16)",
    "garodia_unet_k0": "Garodia U-Net (2D)",
    "garodia_unet_k3": "Garodia U-Net (2.5D, 7 slices)",
    "segformer_k0": "SegFormer-B2 (2D, ImageNet)",
    "unetpp_r34_k0": "UNet++ ResNet-34 (2D, ImageNet)",
    "transunet_k0": "TransUNet R50-ViT-B/16 (2D, ImageNet-21k)",
    "resunet_v1labels_k0": "ResUNet trained on v1 labels",
    "millnet_k0": "MillNet (2D) · on the website",
    "millnet_k3": "MillNet (2.5D, 7 slices)",
    "millnet_k3_s1": "MillNet (2.5D, 7 slices, seed 1)",
    "millnet_k3_nobank": "MillNet (2.5D) without streak bank",
    "millnet_k3_nostats": "MillNet (2.5D) without histogram FiLM",
    "millnet_k0_wide": "MillNet (2D), baseline augmentation",
    "millnet_k0_headonly": "MillNet (2D), factorised head only",
    "millnet_k0_noaxial": "MillNet (2D), head + FiLM, no attention",
    "resunet_k0_s1": "ResUNet (Éponge, 2D, seed 1)",
    "resunet_k0_narrowscale": "ResUNet (2D), MillNet augmentation",
    "garodia_unet_k3_s1": "Garodia U-Net (2.5D, seed 1)",
    "previous_2d_unet": "Previous 2D U-Net (v1 labels, earlier experiment)",
    "baseline_threshold_140": "Threshold I < 140*",
    "baseline_otsu_raw": "Otsu on raw image*",
    "baseline_local_threshold": "Local threshold*",
}


def previous_model_predict(stack, device, ckpt_path):
    from legacy_unet import UNet as TUNet  # architecture of the earlier experiment (eponge/scripts/legacy_unet.py)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = TUNet(1, 3, base_ch=32, bilinear=True)
    m.load_state_dict(ck["model"]); m.to(device).eval()
    out = {}
    with torch.no_grad():
        for z in TEST:  # its own inference convention: raw / 255, full frame
            x = torch.from_numpy(stack[z].astype(np.float32) / 255.0)[None, None]
            H, W = x.shape[-2:]
            xp = torch.nn.functional.pad(x, (0, (-W) % 16, 0, (-H) % 16), mode="reflect").to(device)
            out[z] = m(xp)[..., :H, :W].argmax(1)[0].cpu().numpy().astype(np.uint8)
    return out, sum(p.numel() for p in m.parameters())


def garodia_scores(preds, trimaps):
    """Pore IoU / accuracy on Garodia's confident pixels, and whether porosity falls inside his low-high bracket."""
    inter = union = correct = total = 0
    inside = []
    for z, p in preds.items():
        t = trimaps[z]
        conf = (t == 0) | (t == 255)
        pp, tp = p == 1, t == 0
        inter += (pp & tp & conf).sum(); union += ((pp | tp) & conf).sum()
        correct += ((pp == tp) & conf).sum(); total += conf.sum()
        region = t != 128
        lo = (t[region] == 0).mean(); hi = ((t[region] == 0) | (t[region] == 64)).mean()
        roi = p != 2
        por = (p[roi] == 1).mean() if roi.any() else np.nan
        inside.append(lo <= por <= hi)
    return {"pore_iou_confident": float(inter / max(union, 1)), "accuracy_confident": float(correct / max(total, 1)),
            "porosity_inside_bracket": float(np.mean(inside))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--previous-ckpt", default=None, help="earlier experiment's three_class_real_finetune/best.pt")
    a = ap.parse_args()
    stack = load_stack(ROOT / "data/raw/comp_full.tif", range(125))
    v2 = {z: export_to_classes(m) for z, m in load_masks(ROOT / "data/annotations_v2").items()}
    v1 = {z: export_to_classes(m) for z, m in load_masks(ROOT / "data/manual_annotation").items()}
    gdir = ROOT / "outputs/comp/manual_borders/trimap"
    trimaps = load_masks(gdir) if gdir.exists() else {}

    preds, info = {}, {}
    for ckpt in sorted((ROOT / "runs").glob("*/best.pt")):
        name = ckpt.parent.name
        if name not in NAMES:
            continue
        model, ck = load_checkpoint(ckpt, a.device)
        cls, _ = segment_stack(model, stack, ck.get("k", 0), a.device, tta=True, frames=TEST)
        preds[name] = dict(zip(TEST, cls))
        info[name] = {"params_M": count_parameters(model) / 1e6, "arch": ck["config"].get("arch", "resunet"),
                      "k": ck.get("k", 0), "best_iter": ck.get("iter")}
        print("predicted", name, flush=True)
        del model
    if a.previous_ckpt:
        preds["previous_2d_unet"], n = previous_model_predict(stack, a.device, a.previous_ckpt)
        info["previous_2d_unet"] = {"params_M": n / 1e6, "arch": "unet (earlier)", "k": 0}
    for name, fn in {"baseline_threshold_140": fixed_threshold, "baseline_otsu_raw": otsu_raw,
                     "baseline_local_threshold": local_threshold}.items():
        preds[name] = {z: fn(stack[z], v2[z] != 2) for z in TEST}
        info[name] = {"note": "given the reference catalyst-layer outline (oracle ROI)"}

    table = {}
    for name, p in preds.items():
        r2, r1 = evaluate_frames(p, {z: v2[z] for z in TEST}), evaluate_frames(p, {z: v1[z] for z in TEST})
        row = {**info.get(name, {}),
               "v2": {k: r2[k] for k in ("mean_iou", "pore_iou", "solid_iou", "outside_roi_iou", "roi_iou", "pore_f1_tol1",
                                          "porosity_mae", "pred_porosity_mean", "gt_porosity_mean", "z_consistency_pore_dice")},
               "v1": {k: r1[k] for k in ("mean_iou", "pore_iou", "roi_iou", "gt_porosity_mean")}}
        if trimaps:
            row["garodia"] = garodia_scores(p, {z: trimaps[z] for z in TEST})
        table[name] = row
    (ROOT / "runs/comparison.json").write_text(json.dumps(table, indent=2))

    write_reports(table)


def write_reports(table):
    nets = sorted([n for n in table if not n.startswith("baseline")], key=lambda n: -table[n]["v2"]["mean_iou"])
    order = nets + sorted([n for n in table if n.startswith("baseline")], key=lambda n: -table[n]["v2"]["mean_iou"])
    lines = ["| model | params | mIoU | pore IoU | layer IoU | pore F1 ±1px | porosity error | z-consistency | "
             "pore IoU vs Garodia trimap | in Garodia bracket |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n_ in order:
        if n_ == order[len(nets)] if len(order) > len(nets) else False:
            lines.append("| *Classical baselines (given the true layer outline)* | | | | | | | | | |")
        r = table[n_]; v = r["v2"]; g = r.get("garodia", {})
        pm = f"{r['params_M']:.1f} M" if "params_M" in r else "–"
        lines.append(f"| {NAMES[n_]} | {pm} | {v['mean_iou']:.3f} | {v['pore_iou']:.3f} | {v['roi_iou']:.3f} | "
                     f"{v['pore_f1_tol1']:.3f} | {v['porosity_mae']:.3f} | {v['z_consistency_pore_dice']:.3f} | "
                     f"{g.get('pore_iou_confident', float('nan')):.3f} | {100 * g.get('porosity_inside_bracket', float('nan')):.0f} % |")
    md = "\n".join(lines)
    ref = table[order[0]]["v2"]["gt_porosity_mean"]
    md += (f"\n\nTest frames 100–117 (never used for training or checkpoint selection). Reference porosity "
           f"(v2): {ref:.3f}. * Baselines are given the reference catalyst-layer outline (layer IoU 1.000); the networks "
           "find the layer themselves. Raw Otsu scores high on mIoU because the v2 labels are themselves a denoised Otsu "
           "rule; its boundary noise shows in the lowest z-consistency. z-consistency: pore Dice between neighbouring "
           "predicted slices (labels themselves: 0.855 v2, 0.764 v1 on frames 61–90). Garodia trimap: D. Garodia's "
           "independent Otsu + Ferner-style consensus, scored on its confident pixels; 'in bracket' = share of test frames "
           "whose predicted porosity lies between his low and high estimates.\n")
    (ROOT / "docs/model_comparison.md").write_text(md)
    print(md)

    best = nets[0]
    web = ["millnet_k0", "millnet_k3", "resunet_k0", "garodia_unet_k3", "unetpp_r34_k0", "transunet_k0", "segformer_k0",
           "resunet_v1labels_k0", "previous_2d_unet", "baseline_otsu_raw", "baseline_threshold_140"]
    order = [n_ for n_ in order if n_ in web]   # the website shows the key models; docs keep every run
    rows = [{"name": NAMES[n_], "miou": table[n_]["v2"]["mean_iou"], "pore_iou": table[n_]["v2"]["pore_iou"],
             "roi_iou": table[n_]["v2"]["roi_iou"], "por_mae": table[n_]["v2"]["porosity_mae"],
             "best": n_ == "millnet_k0"} for n_ in order]
    (ROOT / "webapp/model/results.json").write_text(json.dumps({
        "rows": rows, "best_overall": NAMES[best],
        "note": "* Threshold baselines are handed the true catalyst-layer outline; the networks find it themselves. "
                "|Δε| is the mean absolute error of per-frame porosity. Frames 100–117 were never used for training "
                "or checkpoint selection.",
        "footer": f"Test-set reference porosity {ref:.3f} (corrected labels)"}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--from-json":   # regenerate reports without re-running inference
        write_reports(json.loads((ROOT / "runs/comparison.json").read_text()))
    else:
        main()
