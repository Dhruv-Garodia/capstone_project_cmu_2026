"""Segmentation metrics against reference labels."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .io import CLASS_NAMES

NC = 3


def confusion(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.bincount(target.ravel().astype(np.int64) * NC + pred.ravel(), minlength=NC * NC).reshape(NC, NC)


def from_confusion(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    res = {"pixel_accuracy": float(np.trace(cm) / max(cm.sum(), 1))}
    ious = []
    for i, n in enumerate(CLASS_NAMES):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i].sum() - tp
        res[f"{n}_iou"] = float(tp / max(tp + fp + fn, 1))
        res[f"{n}_dice"] = float(2 * tp / max(2 * tp + fp + fn, 1))
        ious.append(res[f"{n}_iou"])
    res["mean_iou"] = float(np.mean(ious))
    inter = cm[:2, :2].sum()
    res["roi_iou"] = float(inter / max(cm[:2].sum() + cm[:, :2].sum() - inter, 1))
    return res


def boundary_tolerant_pore_f1(pred: np.ndarray, target: np.ndarray, tol: int = 1) -> float:
    """Pore F1 where a pixel within ``tol`` px of the reference boundary counts as correct either way."""
    gp, pp = target == 1, pred == 1
    band = ndi.binary_dilation(gp, iterations=tol) & ~ndi.binary_erosion(gp, iterations=tol)
    valid = (target != 2) & ~band
    tp = (gp & pp & valid).sum(); fp = (~gp & pp & valid).sum(); fn = (gp & ~pp & valid).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def evaluate_frames(preds: dict[int, np.ndarray], targets: dict[int, np.ndarray]) -> dict:
    cm = np.zeros((NC, NC), np.int64)
    rows = []
    for z in sorted(preds):
        p, t = preds[z], targets[z]
        cm += confusion(p, t)
        gt_por = float((t[t != 2] == 1).mean())
        pr_por = float((p[p != 2] == 1).mean()) if (p != 2).any() else 0.0
        rows.append({"frame": z, "gt_porosity": gt_por, "pred_porosity": pr_por,
                     "pore_f1_tol1": boundary_tolerant_pore_f1(p, t)})
    res = from_confusion(cm)
    res["confusion_rows_true_cols_pred"] = cm.tolist()
    res["porosity_mae"] = float(np.mean([abs(r["gt_porosity"] - r["pred_porosity"]) for r in rows]))
    res["pred_porosity_mean"] = float(np.mean([r["pred_porosity"] for r in rows]))
    res["gt_porosity_mean"] = float(np.mean([r["gt_porosity"] for r in rows]))
    res["pore_f1_tol1"] = float(np.mean([r["pore_f1_tol1"] for r in rows]))
    zs = sorted(preds)
    zd = [2 * ((preds[a] == 1) & (preds[b] == 1)).sum() / max((preds[a] == 1).sum() + (preds[b] == 1).sum(), 1)
          for a, b in zip(zs[:-1], zs[1:]) if b == a + 1]
    res["z_consistency_pore_dice"] = float(np.mean(zd)) if zd else float("nan")
    res["per_frame"] = rows
    return res
