"""Review / hand-edit the corrected annotations in napari.

    python scripts/napari_review.py --stack ../data/raw/comp_full.tif \
        --v1 ../data/manual_annotation --v2 ../data/annotations_v2 [--pred ../runs/unet2d_v2/test_predictions]

Layers: raw image, v1 labels, v2 labels (editable: 1 = pore, 2 = solid, 3 = outside),
"changed" (v1 != v2), and optional model predictions. Press **Shift+S** to save the edited
v2 layer back to 0/128/255 PNGs (originals are kept in ``<v2>/_before_edit``).
"""

import argparse
import shutil
from pathlib import Path

import napari
import numpy as np

from eponge.io import load_masks, load_stack, save_mask

# napari label ids (0 is transparent in napari, so classes are shifted)
EXPORT_TO_NAPARI = {0: 1, 255: 2, 128: 3}


def to_napari(masks: dict, n: int, shape) -> np.ndarray:
    vol = np.zeros((n,) + shape, np.uint8)
    for z, m in masks.items():
        if z < n:
            for v, lab in EXPORT_TO_NAPARI.items():
                vol[z][m == v] = lab
    return vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True); ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True); ap.add_argument("--pred", default=None)
    ap.add_argument("--frames", type=int, default=118)
    a = ap.parse_args()
    stack = load_stack(a.stack, range(a.frames))
    shape = stack.shape[1:]
    v1 = to_napari(load_masks(a.v1), a.frames, shape)
    v2_masks = load_masks(a.v2)
    v2 = to_napari(v2_masks, a.frames, shape)
    viewer = napari.Viewer(title="Eponge annotation review")
    viewer.add_image(stack, name="pFIB-SEM", colormap="gray")
    colors = {None: (0, 0, 0, 0), 0: (0, 0, 0, 0), 1: (0.1, 0.45, 1.0, 1), 2: (0.95, 0.85, 0.3, 1), 3: (0.3, 0.3, 0.3, 1)}
    from napari.utils import DirectLabelColormap
    cmap = DirectLabelColormap(color_dict=colors)
    viewer.add_labels(v1, name="v1 (napari, threshold-based)", opacity=0.35, colormap=cmap, visible=False)
    lay = viewer.add_labels(v2, name="v2 corrected (editable)", opacity=0.35, colormap=cmap)
    viewer.add_labels(((v1 != v2) & (v1 > 0)).astype(np.uint8), name="changed v1->v2", opacity=0.6, visible=False)
    if a.pred:
        viewer.add_labels(to_napari(load_masks(a.pred), a.frames, shape), name="model prediction",
                          opacity=0.35, colormap=cmap, visible=False)
    lay.selected_label = 1
    lay.brush_size = 6

    @viewer.bind_key("Shift-S")
    def _save(v):
        out = Path(a.v2)
        backup = out / "_before_edit"
        backup.mkdir(exist_ok=True)
        data = np.asarray(lay.data)
        for z in v2_masks:
            src = out / f"slice_{z:04d}.png"
            if src.exists() and not (backup / src.name).exists():
                shutil.copy(src, backup / src.name)
            cls = np.full(shape, 2, np.uint8)
            cls[data[z] == 1] = 1
            cls[data[z] == 2] = 0
            save_mask(src, cls)
        print(f"saved {len(v2_masks)} masks to {out}")

    napari.run()


if __name__ == "__main__":
    main()
