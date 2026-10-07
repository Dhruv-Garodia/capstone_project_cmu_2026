"""Finalize three-class real-data masks with deterministic 2.5D boundary rules.

Pore and solid are both material. Ignore is reserved for regions outside the
material and must remain connected to an image border.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
from scipy import ndimage
import tifffile


PORE, OUTSIDE, SOLID = 0, 128, 255


def parse_args():
    p = argparse.ArgumentParser(description="Regularize real-data material boundaries across z")
    p.add_argument("--stack", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--edge-search", type=int, default=3)
    p.add_argument("--confidence-boundary-width", type=int, default=3)
    p.add_argument("--panel-count", type=int, default=16)
    p.add_argument("--no-gradient-snap", action="store_true")
    p.add_argument("--no-indentation-repair", action="store_true")
    p.add_argument("--indent-window", type=int, default=41)
    p.add_argument("--indent-min-depth", type=float, default=8.0)
    p.add_argument("--indent-max-width", type=int, default=32)
    p.add_argument("--indent-edge-quantile", type=float, default=0.75)
    p.add_argument("--indent-max-edge-support", type=float, default=0.35)
    p.add_argument("--indent-temporal-support", type=int, default=3)
    p.add_argument("--indent-x-tolerance", type=int, default=3)
    p.add_argument("--no-surface-smoothing", action="store_true")
    p.add_argument("--smooth-window", type=int, default=9)
    p.add_argument("--smooth-min-deviation", type=float, default=2.0)
    p.add_argument("--smooth-max-width", type=int, default=16)
    p.add_argument("--smooth-sigma", type=float, default=1.25)
    p.add_argument("--smooth-temporal-weight", type=float, default=0.35)
    p.add_argument("--no-curve-smoothing", action="store_true")
    p.add_argument("--curve-sigma", type=float, default=4.0)
    p.add_argument("--curve-temporal-weight", type=float, default=0.15)
    p.add_argument("--curve-max-temporal-shift", type=float, default=2.0)
    return p.parse_args()


def discover_masks(directory: Path) -> dict[int, Path]:
    result = {}
    for path in sorted(directory.glob("*.png")):
        tail = path.stem.rsplit("_", 1)[-1]
        if not tail.isdigit():
            continue
        z = int(tail)
        if z in result:
            raise ValueError(f"Duplicate frame {z}: {result[z]} and {path}")
        result[z] = path
    if not result:
        raise ValueError(f"No masks found in {directory}")
    return result


def load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    unexpected = set(np.unique(mask).tolist()) - {PORE, OUTSIDE, SOLID}
    if unexpected:
        raise ValueError(f"Unexpected values in {path}: {sorted(unexpected)}")
    return mask


def build_split(frames: list[int], radius: int) -> dict[str, list[int]]:
    test_blocks = ((10, 12), (40, 42), (70, 72), (100, 102))
    val_blocks = ((20, 22), (50, 52), (80, 82), (110, 112))
    expand = lambda blocks: {z for lo, hi in blocks for z in range(lo, hi + 1)}
    available = set(frames)
    test = expand(test_blocks) & available
    val = expand(val_blocks) & available
    held = test | val
    buffer = {q for z in held for q in range(z - radius, z + radius + 1)} - held
    buffer &= available
    train = available - held - buffer
    return {
        "train_frames": sorted(train), "val_frames": sorted(val),
        "test_frames": sorted(test), "buffer_frames": sorted(buffer),
    }


def surfaces(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not roi.any(axis=0).all():
        raise ValueError("Every image column must intersect the material ROI")
    top = np.argmax(roi, axis=0).astype(np.float32)
    bottom = (roi.shape[0] - 1 - np.argmax(roi[::-1], axis=0)).astype(np.float32)
    return top, bottom


def robust_scale(values: np.ndarray) -> tuple[float, float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad, 1.4826 * mad


def interpolate(frames: list[int], values: np.ndarray, depth: int) -> np.ndarray:
    known = np.asarray(frames)
    all_z = np.arange(depth)
    output = np.empty((depth, values.shape[1]), dtype=np.float32)
    for x in range(values.shape[1]):
        output[:, x] = np.interp(all_z, known, values[:, x])
    return output


def regularize_surfaces(frames, values, depth, radius):
    full = interpolate(frames, values, depth)
    adjacent = [np.abs(values[i] - values[i - 1]) for i in range(1, len(frames))
                if frames[i] - frames[i - 1] == 1]
    adjacent = np.concatenate(adjacent) if adjacent else np.array([0.0])
    med, mad, sigma = robust_scale(adjacent)
    outlier_threshold = med + 3 * max(sigma, 1.0)
    agreement_threshold = med + 2 * max(sigma, 1.0)
    corrected = values.copy()
    candidates = np.zeros_like(values, dtype=bool)
    for i, z in enumerate(frames):
        window = full[max(0, z-radius):min(depth, z+radius+1)]
        consensus = np.median(window, axis=0)
        need = min(3, len(window))
        consensus_support = (np.abs(window - consensus) <= agreement_threshold).sum(axis=0)
        current_support = (np.abs(window - values[i]) <= agreement_threshold).sum(axis=0)
        candidate = ((np.abs(values[i] - consensus) > outlier_threshold)
                     & (consensus_support >= need) & (current_support < need))
        candidate = ndimage.binary_closing(candidate, structure=np.ones(3, bool))
        corrected[i, candidate] = consensus[candidate]
        candidates[i] = candidate
    return corrected, candidates, {
        "adjacent_delta_median_px": med, "adjacent_delta_mad_px": mad,
        "adjacent_delta_sigma_px": sigma, "outlier_threshold_px": outlier_threshold,
        "agreement_threshold_px": agreement_threshold,
    }


def snap_to_gradient(image, proposed, original, candidate, search):
    if not candidate.any() or search <= 0:
        return proposed, 0
    gradient = np.abs(ndimage.sobel(image.astype(np.float32), axis=0, mode="nearest"))
    height, width = image.shape
    xs = np.arange(width)
    original_y = np.clip(np.rint(original).astype(int), 0, height - 1)
    min_strength = float(np.quantile(gradient[original_y, xs], 0.25))
    output = proposed.copy()
    accepted = 0
    for x in np.where(candidate)[0]:
        center = int(np.clip(round(float(proposed[x])), 0, height - 1))
        lo, hi = max(0, center-search), min(height, center+search+1)
        best = lo + int(np.argmax(gradient[lo:hi, x]))
        if gradient[best, x] >= min_strength:
            output[x] = best
            accepted += 1
    return output, accepted


def _validate_indentation_parameters(
    spatial_window,
    min_depth,
    max_width,
    edge_quantile,
    max_edge_support,
    temporal_support,
    x_tolerance,
):
    if spatial_window < 3 or spatial_window % 2 == 0:
        raise ValueError("indent-window must be an odd integer >= 3")
    if min_depth <= 0:
        raise ValueError("indent-min-depth must be positive")
    if max_width < 1:
        raise ValueError("indent-max-width must be >= 1")
    if not 0.0 <= edge_quantile <= 1.0:
        raise ValueError("indent-edge-quantile must be in [0, 1]")
    if not 0.0 <= max_edge_support <= 1.0:
        raise ValueError("indent-max-edge-support must be in [0, 1]")
    if temporal_support < 1:
        raise ValueError("indent-temporal-support must be >= 1")
    if x_tolerance < 0:
        raise ValueError("indent-x-tolerance must be >= 0")


def _surface_gradient_strength(image, surface):
    image = image.astype(np.float32)
    gx = ndimage.sobel(image, axis=1, mode="nearest")
    gy = ndimage.sobel(image, axis=0, mode="nearest")
    magnitude = np.hypot(gx, gy)
    y = np.clip(np.rint(surface).astype(int), 0, image.shape[0] - 1)
    return magnitude[y, np.arange(image.shape[1])]


def _spatial_indentation_masks(values, side, spatial_window, min_depth):
    baseline = ndimage.median_filter(
        values, size=(1, spatial_window), mode="nearest"
    )
    baseline = ndimage.gaussian_filter1d(
        baseline,
        sigma=max(1.0, spatial_window / 10.0),
        axis=1,
        mode="nearest",
    )
    if side == "top":
        inward_depth = values - baseline
    elif side == "bottom":
        inward_depth = baseline - values
    else:
        raise ValueError(f"Unknown surface side: {side}")
    core = inward_depth >= min_depth
    region = inward_depth >= max(1.0, min_depth * 0.25)
    return baseline, inward_depth, core, region


def _temporal_component_support(
    frames, core, frame_index, x_start, x_stop, radius, x_tolerance
):
    z = frames[frame_index]
    left = max(0, x_start - x_tolerance)
    right = min(core.shape[1], x_stop + x_tolerance)
    return sum(
        bool(core[j, left:right].any())
        for j, other_z in enumerate(frames)
        if abs(other_z - z) <= radius
    )


def repair_unsupported_indentations(
    stack,
    frames,
    values,
    side,
    radius=2,
    spatial_window=41,
    min_depth=8.0,
    max_width=32,
    edge_quantile=0.75,
    max_edge_support=0.35,
    temporal_support=3,
    x_tolerance=3,
):
    """Remove only deep, narrow, weak-edge indentations lacking z support.

    Border connectivity alone is not sufficient evidence that a narrow intrusion is
    a physical exterior hole. Wide holes, strong image edges, and structures
    supported by ``temporal_support`` slices are preserved.
    """
    _validate_indentation_parameters(
        spatial_window,
        min_depth,
        max_width,
        edge_quantile,
        max_edge_support,
        temporal_support,
        x_tolerance,
    )
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(frames), stack.shape[2]):
        raise ValueError(
            f"Surface shape {values.shape} does not match "
            f"{len(frames)} frames and width {stack.shape[2]}"
        )

    baseline, inward_depth, core, region = _spatial_indentation_masks(
        values, side, spatial_window, min_depth
    )
    corrected = values.copy()
    accepted = np.zeros_like(core)
    totals = {
        "candidate_segments": 0,
        "accepted_segments": 0,
        "accepted_columns": 0,
        "rejected_too_wide": 0,
        "rejected_image_edge": 0,
        "rejected_temporal_support": 0,
    }
    per_frame = []

    for i, z in enumerate(frames):
        labels, count = ndimage.label(region[i], structure=np.ones(3, bool))
        edge_strength = _surface_gradient_strength(stack[z], values[i])
        positive_edges = edge_strength[edge_strength > 0]
        strong_edge = (
            float(np.quantile(positive_edges, edge_quantile))
            if positive_edges.size
            else float("inf")
        )
        frame_stats = {key: 0 for key in totals}

        for label_id in range(1, count + 1):
            columns = np.flatnonzero(labels == label_id)
            if columns.size == 0 or not core[i, columns].any():
                continue
            totals["candidate_segments"] += 1
            frame_stats["candidate_segments"] += 1
            width = int(columns[-1] - columns[0] + 1)
            if width > max_width:
                totals["rejected_too_wide"] += 1
                frame_stats["rejected_too_wide"] += 1
                continue

            support = _temporal_component_support(
                frames,
                core,
                i,
                int(columns[0]),
                int(columns[-1] + 1),
                radius,
                x_tolerance,
            )
            if support >= temporal_support:
                totals["rejected_temporal_support"] += 1
                frame_stats["rejected_temporal_support"] += 1
                continue

            edge_fraction = float(np.mean(edge_strength[columns] >= strong_edge))
            if edge_fraction > max_edge_support:
                totals["rejected_image_edge"] += 1
                frame_stats["rejected_image_edge"] += 1
                continue

            if side == "top":
                corrected[i, columns] = np.minimum(
                    corrected[i, columns], baseline[i, columns]
                )
            else:
                corrected[i, columns] = np.maximum(
                    corrected[i, columns], baseline[i, columns]
                )
            accepted[i, columns] = True
            totals["accepted_segments"] += 1
            totals["accepted_columns"] += int(columns.size)
            frame_stats["accepted_segments"] += 1
            frame_stats["accepted_columns"] += int(columns.size)

        per_frame.append({"frame": z, **frame_stats})

    audit = {
        "side": side,
        "config": {
            "radius": radius,
            "spatial_window": spatial_window,
            "min_depth": min_depth,
            "max_width": max_width,
            "edge_quantile": edge_quantile,
            "max_edge_support": max_edge_support,
            "temporal_support": temporal_support,
            "x_tolerance": x_tolerance,
        },
        **totals,
        "per_frame": per_frame,
        "max_candidate_depth": float(inward_depth[core].max()) if core.any() else 0.0,
    }
    return corrected, accepted, audit


def smooth_jagged_surfaces(
    frames,
    values,
    depth,
    radius=2,
    spatial_window=9,
    min_deviation=2.0,
    max_width=16,
    sigma=1.25,
    temporal_weight=0.35,
):
    """Remove narrow surface spikes without flattening broad exterior cavities.

    A short horizontal median supplies the local shape prior. Its temporally
    median-filtered counterpart supplies z continuity. Only narrow components
    that depart from the local prior are replaced; broad holes remain unchanged.
    """
    if spatial_window < 3 or spatial_window % 2 == 0:
        raise ValueError("smooth-window must be an odd integer >= 3")
    if min_deviation <= 0:
        raise ValueError("smooth-min-deviation must be positive")
    if max_width < 1:
        raise ValueError("smooth-max-width must be >= 1")
    if sigma < 0:
        raise ValueError("smooth-sigma must be non-negative")
    if not 0.0 <= temporal_weight <= 1.0:
        raise ValueError("smooth-temporal-weight must be in [0, 1]")

    values = np.asarray(values, dtype=np.float32)
    spatial = ndimage.median_filter(
        values, size=(1, spatial_window), mode="nearest"
    )
    full_spatial = interpolate(frames, spatial, depth)
    temporal = np.empty_like(spatial)
    for i, z in enumerate(frames):
        window = full_spatial[max(0, z-radius):min(depth, z+radius+1)]
        temporal[i] = np.median(window, axis=0)

    # Temporal consensus regularizes z continuity but must not recreate a
    # narrow x-direction spike that the local shape prior has rejected.
    temporal_delta = np.clip(
        temporal - spatial, -min_deviation, min_deviation
    )
    target = spatial + temporal_weight * temporal_delta
    if sigma > 0:
        target = ndimage.gaussian_filter1d(
            target, sigma=sigma, axis=1, mode="nearest"
        )

    deviation = np.abs(values - spatial)
    core = deviation >= min_deviation
    shoulder = deviation >= max(0.5, min_deviation * 0.25)
    corrected = values.copy()
    accepted = np.zeros_like(core)
    per_frame = []
    totals = {
        "candidate_segments": 0,
        "accepted_segments": 0,
        "accepted_columns": 0,
        "rejected_too_wide": 0,
    }

    for i, z in enumerate(frames):
        labels, count = ndimage.label(shoulder[i], structure=np.ones(3, bool))
        frame_stats = {key: 0 for key in totals}
        for label_id in range(1, count + 1):
            columns = np.flatnonzero(labels == label_id)
            if columns.size == 0 or not core[i, columns].any():
                continue
            totals["candidate_segments"] += 1
            frame_stats["candidate_segments"] += 1
            width = int(columns[-1] - columns[0] + 1)
            if width > max_width:
                totals["rejected_too_wide"] += 1
                frame_stats["rejected_too_wide"] += 1
                continue

            corrected[i, columns] = target[i, columns]
            accepted[i, columns] = True
            totals["accepted_segments"] += 1
            totals["accepted_columns"] += int(columns.size)
            frame_stats["accepted_segments"] += 1
            frame_stats["accepted_columns"] += int(columns.size)
        per_frame.append({"frame": z, **frame_stats})

    audit = {
        "enabled": True,
        "config": {
            "radius": radius,
            "spatial_window": spatial_window,
            "min_deviation": min_deviation,
            "max_width": max_width,
            "sigma": sigma,
            "temporal_weight": temporal_weight,
        },
        **totals,
        "max_candidate_deviation": (
            float(deviation[core].max()) if core.any() else 0.0
        ),
        "per_frame": per_frame,
    }
    return corrected, accepted, audit


def smooth_continuous_surfaces(
    frames,
    values,
    depth,
    radius=2,
    sigma=4.0,
    temporal_weight=0.15,
    max_temporal_shift=2.0,
):
    """Round surface corners while preserving broad boundary-hole geometry."""
    if sigma < 0:
        raise ValueError("curve-sigma must be non-negative")
    if not 0.0 <= temporal_weight <= 1.0:
        raise ValueError("curve-temporal-weight must be in [0, 1]")
    if max_temporal_shift < 0:
        raise ValueError("curve-max-temporal-shift must be non-negative")

    values = np.asarray(values, dtype=np.float32)
    horizontal = ndimage.gaussian_filter1d(
        values, sigma=sigma, axis=1, mode="nearest"
    ) if sigma > 0 else values.copy()
    full = interpolate(frames, horizontal, depth)
    temporal = np.empty_like(horizontal)
    for i, z in enumerate(frames):
        window = full[max(0, z-radius):min(depth, z+radius+1)]
        temporal[i] = np.median(window, axis=0)

    temporal_delta = np.clip(
        temporal - horizontal, -max_temporal_shift, max_temporal_shift
    )
    corrected = horizontal + temporal_weight * temporal_delta
    changed = np.abs(corrected - values) >= 0.5
    audit = {
        "enabled": True,
        "config": {
            "radius": radius,
            "sigma": sigma,
            "temporal_weight": temporal_weight,
            "max_temporal_shift": max_temporal_shift,
        },
        "adjusted_columns": int(changed.sum()),
        "max_adjustment": float(np.abs(corrected - values).max()),
        "mean_adjustment": float(np.abs(corrected - values).mean()),
        "per_frame": [
            {"frame": z, "adjusted_columns": int(changed[i].sum())}
            for i, z in enumerate(frames)
        ],
    }
    return corrected, changed, audit


def apply_surfaces(roi, old_top, old_bottom, new_top, new_bottom):
    output = roi.copy()
    height, width = output.shape
    for x in range(width):
        ot, ob, nt, nb = [int(np.clip(round(float(v)), 0, height-1)) for v in
                          (old_top[x], old_bottom[x], new_top[x], new_bottom[x])]
        if nt < ot: output[nt:ot, x] = True
        elif nt > ot: output[ot:nt, x] = False
        if nb > ob: output[ob+1:nb+1, x] = True
        elif nb < ob: output[nb+1:ob+1, x] = False
    return output


def fit_threshold(stack, masks, train_frames):
    pore_hist = np.zeros(256, np.int64)
    solid_hist = np.zeros(256, np.int64)
    for z in train_frames:
        image, mask = np.asarray(stack[z], np.uint8), masks[z]
        pore_hist += np.bincount(image[mask == PORE], minlength=256)
        solid_hist += np.bincount(image[mask == SOLID], minlength=256)
    best = None
    for threshold in range(1, 256):
        for dark in (True, False):
            if dark:
                tp, fn, fp, tn = (pore_hist[:threshold].sum(), pore_hist[threshold:].sum(),
                                  solid_hist[:threshold].sum(), solid_hist[threshold:].sum())
            else:
                tp, fn, fp, tn = (pore_hist[threshold:].sum(), pore_hist[:threshold].sum(),
                                  solid_hist[threshold:].sum(), solid_hist[:threshold].sum())
            bal = (tp/max(tp+fn, 1) + tn/max(tn+fp, 1)) / 2
            iou = tp/max(tp+fp+fn, 1)
            row = (float(bal), float(iou), threshold, dark)
            if best is None or row[:2] > best[:2]: best = row
    return {"balanced_accuracy": best[0], "pore_iou": best[1],
            "threshold": best[2], "pore_is_dark": best[3]}


def merge_phase(image, original, corrected_roi, threshold_info):
    output = original.copy()
    old_roi = original != OUTSIDE
    output[old_roi & ~corrected_roi] = OUTSIDE
    added = ~old_roi & corrected_roi
    threshold = threshold_info["threshold"]
    pore = image < threshold if threshold_info["pore_is_dark"] else image >= threshold
    output[added & pore] = PORE
    output[added & ~pore] = SOLID
    return output


def fill_enclosed_ignore(image, mask, threshold_info):
    """Reclassify ignore islands that are not connected to the image exterior."""
    outside = mask == OUTSIDE
    seeds = np.zeros_like(outside)
    seeds[0] = outside[0]
    seeds[-1] = outside[-1]
    seeds[:, 0] |= outside[:, 0]
    seeds[:, -1] |= outside[:, -1]
    exterior = ndimage.binary_propagation(
        seeds, structure=np.ones((3, 3), bool), mask=outside
    )
    enclosed = outside & ~exterior
    labels, component_count = ndimage.label(
        enclosed, structure=np.ones((3, 3), bool)
    )
    sizes = np.bincount(labels.ravel())[1:]
    output = mask.copy()
    threshold = threshold_info["threshold"]
    pore = image < threshold if threshold_info["pore_is_dark"] else image >= threshold
    output[enclosed & pore] = PORE
    output[enclosed & ~pore] = SOLID
    return output, enclosed, int(component_count), int(sizes.max()) if sizes.size else 0


def boundary(mask):
    return (ndimage.binary_dilation(mask, np.ones((3, 3), bool))
            ^ ndimage.binary_erosion(mask, np.ones((3, 3), bool)))


def make_confidence(image, old_roi, new_roi, threshold, width, margin):
    uncertain = boundary(old_roi) | boundary(new_roi)
    uncertain = ndimage.binary_dilation(uncertain, np.ones((2*width+1, 2*width+1), bool))
    uncertain |= np.abs(image.astype(np.int16) - threshold) <= margin
    return (~uncertain).astype(np.uint8) * 255


def dice(a, b):
    return float(2*np.logical_and(a, b).sum()/max(a.sum()+b.sum(), 1))


def stats(mask):
    roi = mask != OUTSIDE
    return {"roi_fraction": float(roi.mean()),
            "porosity": float((mask == PORE).sum()/max(roi.sum(), 1))}


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save_panel(image, old, new, confidence, z, path):
    cmap = ListedColormap(["#0aa6b4", "#e07a5f", "#f2f0e6"])
    def cls(mask):
        out = np.zeros_like(mask); out[mask == OUTSIDE] = 1; out[mask == SOLID] = 2
        return out
    changed = old != new
    fig, ax = plt.subplots(1, 5, figsize=(22, 5), constrained_layout=True)
    ax[0].imshow(image, cmap="gray", vmin=0, vmax=255); ax[0].set_title(f"Raw {z}")
    ax[1].imshow(cls(old), cmap=cmap, vmin=0, vmax=2); ax[1].set_title("Original")
    ax[2].imshow(cls(new), cmap=cmap, vmin=0, vmax=2); ax[2].set_title("Corrected")
    ax[3].imshow(image, cmap="gray", vmin=0, vmax=255)
    ax[3].imshow(np.ma.masked_where(~changed, changed), cmap="autumn", alpha=.8)
    ax[3].set_title(f"Changed {changed.mean():.3%}")
    confidence_display = (confidence > 0).astype(np.uint8)
    confidence_cmap = ListedColormap(["#f4a261", "#e9f5db"])
    ax[4].imshow(confidence_display, cmap=confidence_cmap, vmin=0, vmax=1)
    ax[4].set_title("Confidence: amber=uncertain (not ignore)")
    for a in ax: a.axis("off")
    fig.savefig(path, dpi=140); plt.close(fig)


def split_distribution(masks, split):
    result = {}
    for key in ("train_frames", "val_frames", "test_frames", "buffer_frames"):
        counts = np.zeros(3, np.int64)
        for z in split[key]:
            m = masks[z]; counts += [(m == SOLID).sum(), (m == PORE).sum(), (m == OUTSIDE).sum()]
        total, roi = counts.sum(), counts[:2].sum()
        result[key] = {"frame_count": len(split[key]), "frames": split[key],
                       "pixel_counts_solid_pore_outside": counts.tolist(),
                       "pixel_fractions_solid_pore_outside": (counts/max(total, 1)).tolist(),
                       "pore_fraction_within_roi": float(counts[1]/max(roi, 1))}
    return result


def main():
    args = parse_args()
    out = Path(args.output_dir)
    corrected_dir, confidence_dir = out/"corrected_masks", out/"confidence_masks"
    report_dir, panel_dir = out/"mask_correction_report", out/"mask_correction_report"/"panels"
    for p in (corrected_dir, confidence_dir, report_dir, panel_dir): p.mkdir(parents=True, exist_ok=True)
    paths = discover_masks(Path(args.mask_dir)); frames = sorted(paths)
    stack = tifffile.imread(args.stack)
    masks = {z: load_mask(paths[z]) for z in frames}
    shape = next(iter(masks.values())).shape
    if stack.shape[1:] != shape: raise ValueError(f"Shape mismatch: {stack.shape[1:]} vs {shape}")
    split = build_split(frames, args.radius)
    rois = np.stack([masks[z] != OUTSIDE for z in frames])
    surface_pairs = [surfaces(roi) for roi in rois]
    tops = np.stack([p[0] for p in surface_pairs]); bottoms = np.stack([p[1] for p in surface_pairs])
    new_tops, top_candidates, top_meta = regularize_surfaces(frames, tops, stack.shape[0], args.radius)
    new_bottoms, bottom_candidates, bottom_meta = regularize_surfaces(frames, bottoms, stack.shape[0], args.radius)
    snapped = []
    for i, z in enumerate(frames):
        count = 0
        if not args.no_gradient_snap:
            new_tops[i], n = snap_to_gradient(stack[z], new_tops[i], tops[i], top_candidates[i], args.edge_search); count += n
            new_bottoms[i], n = snap_to_gradient(stack[z], new_bottoms[i], bottoms[i], bottom_candidates[i], args.edge_search); count += n
        snapped.append(count)

    indentation_config = {
        "radius": args.radius,
        "spatial_window": args.indent_window,
        "min_depth": args.indent_min_depth,
        "max_width": args.indent_max_width,
        "edge_quantile": args.indent_edge_quantile,
        "max_edge_support": args.indent_max_edge_support,
        "temporal_support": args.indent_temporal_support,
        "x_tolerance": args.indent_x_tolerance,
    }
    if args.no_indentation_repair:
        top_indentation = np.zeros_like(top_candidates)
        bottom_indentation = np.zeros_like(bottom_candidates)
        empty = {
            "enabled": False,
            "config": indentation_config,
            "candidate_segments": 0,
            "accepted_segments": 0,
            "accepted_columns": 0,
            "rejected_too_wide": 0,
            "rejected_image_edge": 0,
            "rejected_temporal_support": 0,
            "per_frame": [{"frame": z} for z in frames],
            "max_candidate_depth": 0.0,
        }
        top_indent_audit = {"side": "top", **empty}
        bottom_indent_audit = {"side": "bottom", **empty}
    else:
        new_tops, top_indentation, top_indent_audit = repair_unsupported_indentations(
            stack, frames, new_tops, "top", **indentation_config
        )
        new_bottoms, bottom_indentation, bottom_indent_audit = repair_unsupported_indentations(
            stack, frames, new_bottoms, "bottom", **indentation_config
        )
        top_indent_audit["enabled"] = True
        bottom_indent_audit["enabled"] = True

    smoothing_config = {
        "radius": args.radius,
        "spatial_window": args.smooth_window,
        "min_deviation": args.smooth_min_deviation,
        "max_width": args.smooth_max_width,
        "sigma": args.smooth_sigma,
        "temporal_weight": args.smooth_temporal_weight,
    }
    if args.no_surface_smoothing:
        top_smoothed = np.zeros_like(top_candidates)
        bottom_smoothed = np.zeros_like(bottom_candidates)
        empty_smoothing = {
            "enabled": False,
            "config": smoothing_config,
            "candidate_segments": 0,
            "accepted_segments": 0,
            "accepted_columns": 0,
            "rejected_too_wide": 0,
            "max_candidate_deviation": 0.0,
            "per_frame": [{"frame": z} for z in frames],
        }
        top_smoothing_audit = {"side": "top", **empty_smoothing}
        bottom_smoothing_audit = {"side": "bottom", **empty_smoothing}
    else:
        new_tops, top_smoothed, top_smoothing_audit = smooth_jagged_surfaces(
            frames, new_tops, stack.shape[0], **smoothing_config
        )
        new_bottoms, bottom_smoothed, bottom_smoothing_audit = smooth_jagged_surfaces(
            frames, new_bottoms, stack.shape[0], **smoothing_config
        )
        top_smoothing_audit["side"] = "top"
        bottom_smoothing_audit["side"] = "bottom"

    curve_config = {
        "radius": args.radius,
        "sigma": args.curve_sigma,
        "temporal_weight": args.curve_temporal_weight,
        "max_temporal_shift": args.curve_max_temporal_shift,
    }
    if args.no_curve_smoothing:
        top_curve_changed = np.zeros_like(top_candidates)
        bottom_curve_changed = np.zeros_like(bottom_candidates)
        empty_curve = {
            "enabled": False,
            "config": curve_config,
            "adjusted_columns": 0,
            "max_adjustment": 0.0,
            "mean_adjustment": 0.0,
            "per_frame": [{"frame": z} for z in frames],
        }
        top_curve_audit = {"side": "top", **empty_curve}
        bottom_curve_audit = {"side": "bottom", **empty_curve}
    else:
        new_tops, top_curve_changed, top_curve_audit = smooth_continuous_surfaces(
            frames, new_tops, stack.shape[0], **curve_config
        )
        new_bottoms, bottom_curve_changed, bottom_curve_audit = smooth_continuous_surfaces(
            frames, new_bottoms, stack.shape[0], **curve_config
        )
        top_curve_audit["side"] = "top"
        bottom_curve_audit["side"] = "bottom"

    threshold_info = fit_threshold(stack, masks, split["train_frames"])
    distances = np.concatenate([np.abs(stack[z].astype(np.int16)-threshold_info["threshold"])[masks[z] != OUTSIDE]
                                for z in split["train_frames"]])
    margin = max(1, int(np.quantile(distances, .10)))
    corrected, confidences, rows = {}, {}, []
    prev_old = prev_new = None; prev_z = None
    for i, z in enumerate(frames):
        old, old_roi = masks[z], rois[i]
        new_roi = apply_surfaces(old_roi, tops[i], bottoms[i], new_tops[i], new_bottoms[i])
        new = merge_phase(stack[z], old, new_roi, threshold_info)
        new, enclosed_filled, enclosed_components, largest_enclosed = fill_enclosed_ignore(
            stack[z], new, threshold_info
        )
        final_roi = new != OUTSIDE
        conf = make_confidence(stack[z], old_roi, final_roi, threshold_info["threshold"],
                               args.confidence_boundary_width, margin)
        corrected[z], confidences[z] = new, conf
        Image.fromarray(new).save(corrected_dir/f"slice_{z:04d}.png")
        Image.fromarray(conf).save(confidence_dir/f"slice_{z:04d}.png")
        old_s, new_s = stats(old), stats(new)
        consecutive = prev_z is not None and z-prev_z == 1
        rows.append({"frame": z, "source_file": paths[z].name,
                     "original_roi_fraction": old_s["roi_fraction"], "corrected_roi_fraction": new_s["roi_fraction"],
                     "original_porosity": old_s["porosity"], "corrected_porosity": new_s["porosity"],
                     "changed_pixels": int((old != new).sum()), "changed_fraction": float((old != new).mean()),
                     "top_columns_corrected": int(top_candidates[i].sum()),
                     "bottom_columns_corrected": int(bottom_candidates[i].sum()),
                     "top_indentation_columns_repaired": int(top_indentation[i].sum()),
                     "bottom_indentation_columns_repaired": int(bottom_indentation[i].sum()),
                     "top_surface_columns_smoothed": int(top_smoothed[i].sum()),
                     "bottom_surface_columns_smoothed": int(bottom_smoothed[i].sum()),
                     "top_curve_columns_adjusted": int(top_curve_changed[i].sum()),
                     "bottom_curve_columns_adjusted": int(bottom_curve_changed[i].sum()),
                     "columns_gradient_snapped": snapped[i],
                     "enclosed_ignore_components_filled": enclosed_components,
                     "enclosed_ignore_pixels_filled": int(enclosed_filled.sum()),
                     "largest_enclosed_ignore_component_filled": largest_enclosed,
                     "high_confidence_fraction": float((conf == 255).mean()),
                     "neighbor_dice_before": dice(prev_old, old_roi) if consecutive else float("nan"),
                     "neighbor_dice_after": dice(prev_new, final_roi) if consecutive else float("nan")})
        prev_old, prev_new, prev_z = old_roi, final_roi, z
    write_csv(report_dir/"per_frame_mask_audit.csv", rows)
    valid = [r for r in rows if np.isfinite(r["neighbor_dice_before"])]
    ranked = sorted(rows, key=lambda r: r["changed_fraction"], reverse=True)
    panel_frames = set(r["frame"] for r in ranked[:args.panel_count]) | ({29,30,31,59,60,61,89,90,93,94} & set(frames))
    for z in sorted(panel_frames): save_panel(stack[z], masks[z], corrected[z], confidences[z], z, panel_dir/f"slice_{z:04d}.png")
    fig, ax = plt.subplots(4, 1, figsize=(13, 13), sharex=True, constrained_layout=True)
    zs = [r["frame"] for r in rows]
    ax[0].plot(zs, [r["original_roi_fraction"] for r in rows], label="original"); ax[0].plot(zs, [r["corrected_roi_fraction"] for r in rows], label="corrected")
    ax[1].plot(zs, [r["original_porosity"] for r in rows]); ax[1].plot(zs, [r["corrected_porosity"] for r in rows])
    ax[2].plot(zs, [r["changed_fraction"] for r in rows], color="#d1495b")
    ax[3].plot(zs, [r["neighbor_dice_before"] for r in rows], label="before"); ax[3].plot(zs, [r["neighbor_dice_after"] for r in rows], label="after")
    labels = ["ROI fraction", "ROI porosity", "Changed fraction", "Previous-slice ROI Dice"]
    for a, label in zip(ax, labels):
        a.set_ylabel(label); [a.axvline(b, color="#555", ls="--", alpha=.5) for b in (30.5,60.5,90.5)]
    ax[0].legend(); ax[3].legend(); ax[3].set_xlabel("Frame")
    fig.savefig(report_dir/"z_consistency_audit.png", dpi=180); plt.close(fig)
    split.update({"strategy": "distributed blocked split across annotator ranges", "context_radius": args.radius,
                  "annotator_ranges": [[0,30],[31,60],[61,90],[91,120]]})
    (out/"split.json").write_text(json.dumps(split, indent=2))
    summary = {"algorithm_version": "v7-final",
               "frame_count": len(frames), "frames": frames, "model_predictions_used": False,
               "method": "Conservative top/bottom material-surface regularization; existing reviewed pore/solid labels are preserved in overlapping ROI.",
               "size_is_not_a_required_external_hole_rule": True,
               "existing_boundary_holes_preserved_when_temporally_supported": True,
               "ignore_topology_rule": "Every ignore component must connect to the image border; enclosed ignore is reclassified as pore/solid.",
               "enclosed_ignore_components_filled": int(sum(r["enclosed_ignore_components_filled"] for r in rows)),
               "enclosed_ignore_pixels_filled": int(sum(r["enclosed_ignore_pixels_filled"] for r in rows)),
               "remaining_enclosed_ignore_pixels": 0,
               "top_surface": top_meta, "bottom_surface": bottom_meta,
               "indentation_repair": {
                   "rule": "Repair only deep, narrow intrusions with weak image-edge evidence and insufficient radius-2 temporal support.",
                   "top": {k: v for k, v in top_indent_audit.items() if k != "per_frame"},
                   "bottom": {k: v for k, v in bottom_indent_audit.items() if k != "per_frame"},
               },
               "surface_smoothing": {
                   "rule": "Replace narrow surface spikes with a short-range spatial prior and bounded radius-2 temporal consensus; preserve broad cavities.",
                   "top": {k: v for k, v in top_smoothing_audit.items() if k != "per_frame"},
                   "bottom": {k: v for k, v in bottom_smoothing_audit.items() if k != "per_frame"},
               },
               "continuous_curve_smoothing": {
                   "rule": "Round sharp corners with a Gaussian x-direction curve prior and at most a bounded 2-pixel temporal-consensus shift.",
                   "top": {k: v for k, v in top_curve_audit.items() if k != "per_frame"},
                   "bottom": {k: v for k, v in bottom_curve_audit.items() if k != "per_frame"},
               },
               "phase_threshold": threshold_info,
               "confidence_intensity_margin": margin,
               "overall_changed_fraction": float(sum(r["changed_pixels"] for r in rows)/(len(rows)*shape[0]*shape[1])),
               "max_frame_changed_fraction": float(max(r["changed_fraction"] for r in rows)),
               "mean_high_confidence_fraction": float(np.mean([r["high_confidence_fraction"] for r in rows])),
               "mean_neighbor_roi_dice_before": float(np.mean([r["neighbor_dice_before"] for r in valid])),
               "mean_neighbor_roi_dice_after": float(np.mean([r["neighbor_dice_after"] for r in valid])),
               "largest_change_frames": [r["frame"] for r in ranked[:10]],
               "split_distribution": split_distribution(corrected, split),
               "interpretation_limit": "Automatically regularized annotations, not independent physical ground truth."}
    (out/"correction_summary.json").write_text(json.dumps(summary, indent=2))
    (report_dir/"README.md").write_text(
        "# Automatic mask correction\n\nModel predictions were not used. Existing pore/solid labels were preserved inside the overlapping ROI. "
        "Size was not used as a mandatory external-hole rule. Existing boundary holes were treated as positive references and preserved when supported by adjacent slices. "
        "Deep, narrow boundary intrusions are repaired only when they have weak image-edge evidence and insufficient radius-2 temporal support. "
        "Narrow surface spikes are smoothed using local shape and radius-2 temporal consensus while broad cavities are preserved. "
        "A continuous-curve prior rounds sharp corners without removing the broad hole geometry. "
        "Ignore must connect to the image border; enclosed ignore islands are reclassified as pore/solid.\n\n"
        f"- Overall changed fraction: {summary['overall_changed_fraction']:.4%}\n"
        f"- Maximum frame changed fraction: {summary['max_frame_changed_fraction']:.4%}\n"
        f"- Neighbor ROI Dice before/after: {summary['mean_neighbor_roi_dice_before']:.6f} / {summary['mean_neighbor_roi_dice_after']:.6f}\n"
        f"- Largest-change frames: {summary['largest_change_frames']}\n\n"
        "These outputs are pseudo-ground-truth, not independent physical ground truth.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
