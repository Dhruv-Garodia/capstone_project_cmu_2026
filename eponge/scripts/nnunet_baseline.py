"""nnU-Net v2 baseline on the final dataset (2D and 3D full-resolution configurations).

nnU-Net "Revisited" (Isensee et al., MICCAI 2024) showed that a properly configured nnU-Net beats most
newer architectures, so it is the baseline a new architecture has to beat. This script

  prepare   writes Dataset501_PFIB in nnU-Net format: each (stack, split) block of consecutive labelled
            slices is one 3D case (labels: 0 outside, 1 solid, 2 pore); train blocks -> training cases,
            val blocks -> the single validation fold (splits_final.json), test blocks -> imagesTs.
  evaluate  converts nnU-Net predictions on the test blocks back to the dataset's class maps and scores
            them with the same evaluate_frames as every other model -> runs/final_models/nnunet_<cfg>/summary.json

    python eponge/scripts/nnunet_baseline.py prepare
    nnUNetv2_plan_and_preprocess -d 501 -c 2d 3d_fullres --verify_dataset_integrity
    (copy splits)  ;  nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_100epochs -device mps
    nnUNetv2_predict -i $nnUNet_raw/Dataset501_PFIB/imagesTs -o OUT -d 501 -c 2d -f 0 -tr nnUNetTrainer_100epochs -device mps
    python eponge/scripts/nnunet_baseline.py evaluate --config 2d --pred OUT
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
import train_final as TF  # noqa: E402
from eponge.evaluate import evaluate_frames  # noqa: E402

BASE = TF.ROOT.parent / "nnunet"
RAW, PRE, RES = BASE / "raw", BASE / "preprocessed", BASE / "results"
DS = "Dataset501_PFIB"
TO_NN = np.array([1, 2, 0], np.uint8)        # dataset classes (0 solid, 1 pore, 2 outside) -> nnU-Net labels
FROM_NN = np.array([2, 0, 1], np.uint8)      # nnU-Net labels -> dataset classes


def blocks(frames):
    """Split a sorted frame list into runs of consecutive frames."""
    out, cur = [], [frames[0]]
    for z in frames[1:]:
        if z == cur[-1] + 1:
            cur.append(z)
        else:
            out.append(cur); cur = [z]
    out.append(cur)
    return out


def write(path, arr):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(path), useCompression=True)


def prepare():
    d = RAW / DS
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    cases = {"train": [], "val": [], "test": []}
    for n in TF.STACKS:
        images, masks, sp = TF.load_stack(n)
        for split in ("train", "val", "test"):
            fr = [z for z in sp[split] if z in masks]
            for b in blocks(fr):
                name = f"{n}_{split}_{b[0]:03d}"
                img = images[b[0]:b[-1] + 1]
                lab = TO_NN[np.stack([masks[z] for z in b])]
                tgt = "Ts" if split == "test" else "Tr"
                write(d / f"images{tgt}" / f"{name}_0000.nii.gz", img)
                write(d / f"labels{tgt}" / f"{name}.nii.gz", lab)
                cases[split].append({"case": name, "stack": n, "frames": b})
    (d / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "SEM"}, "labels": {"background": 0, "solid": 1, "pore": 2},
        "numTraining": len(cases["train"]) + len(cases["val"]), "file_ending": ".nii.gz"}, indent=1))
    (d / "cases.json").write_text(json.dumps(cases, indent=1))
    splits = [{"train": [c["case"] for c in cases["train"]], "val": [c["case"] for c in cases["val"]]}]
    (d / "splits_final.json").write_text(json.dumps(splits, indent=1))
    print({k: len(v) for k, v in cases.items()}, "->", d)


def evaluate(config, pred):
    cases = json.loads((RAW / DS / "cases.json").read_text())["test"]
    test, store = {}, {}
    for n in TF.STACKS:
        _, masks, _ = TF.load_stack(n)
        preds, gts = {}, {}
        for c in [c for c in cases if c["stack"] == n]:
            p = FROM_NN[sitk.GetArrayFromImage(sitk.ReadImage(str(Path(pred) / f"{c['case']}.nii.gz")))]
            for i, z in enumerate(c["frames"]):
                preds[z], gts[z] = p[i], masks[z]
        store[n] = np.stack([preds[z] for z in sorted(preds)]).astype(np.uint8)
        m = evaluate_frames(preds, gts)
        test[n] = {k: m[k] for k in ("mean_iou", "pore_iou", "roi_iou", "pore_f1_tol1", "porosity_mae",
                                     "pred_porosity_mean", "gt_porosity_mean")}
        print(n, {k: round(v, 4) for k, v in test[n].items()}, flush=True)
    out = TF.ROOT.parent / "runs/final_models" / f"nnunet_{config}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "preds.npz", **store)
    (out / "summary.json").write_text(json.dumps({"args": {"arch": f"nnunet_{config}", "train_stacks": "comp,uncomp,pristine",
                                                          "test_stacks": "comp,uncomp,pristine"}, "test": test}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "evaluate"])
    ap.add_argument("--config", default="2d")
    ap.add_argument("--pred", default=None)
    a = ap.parse_args()
    prepare() if a.cmd == "prepare" else evaluate(a.config, a.pred)
