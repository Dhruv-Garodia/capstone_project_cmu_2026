import numpy as np
from skimage import io

# --- Input: your existing binary tif (0=pores, 1=solid) ---
binary_path = "synthetic_like_input.tif"
out_vis      = "synthetic_like_input_vis.tif"     # for human viewing

# Load
vol = io.imread(binary_path)
print("Loaded volume:", vol.shape, vol.dtype, "min/max:", vol.min(), vol.max())

# Option A: pores=black, solid=white
vis = vol.astype(np.uint8) * 255

# Option B (uncomment to invert: pores=white, solid=black)
# vis = (1 - vol.astype(np.uint8)) * 255

# Save
io.imsave(out_vis, vis.astype(np.uint8), check_contrast=False)
print("Saved visualization volume:", out_vis)

# --- Optional: also save a few PNG slices for quick check ---
import os
os.makedirs("slices_preview", exist_ok=True)
mid = vol.shape[0] // 2
io.imsave(f"slices_preview/mid_slice.png", vis[mid].astype(np.uint8))
print("Saved middle slice preview in slices_preview/")
