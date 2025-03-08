import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
def overlay_segmentation(original_file, segmented_file, slice_index=333):
    # Load volumes
    original_stack = tifffile.imread(original_file)
    segmented_stack = tifffile.imread(segmented_file)

    raw_slice = original_stack[slice_index]
    seg_slice = segmented_stack[slice_index]
    
    # If your segmentation is 0 and 255, convert to boolean
    # i.e., True for solid (255), False for pore (0)
    seg_bool = seg_slice > 128

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Show the raw grayscale image
    ax.imshow(raw_slice, cmap='gray')
    
    # Create a masked array where the segmented region is True
    masked_seg = np.ma.masked_where(seg_bool == False, seg_bool)
    
    # Overlay the segmentation in red
    ax.imshow(masked_seg, cmap='autumn', alpha=0.5)
    
    ax.set_title(f"Overlay of Raw and Segmented Slice {slice_index}")
    ax.axis('off')
    plt.show()

if __name__ == "__main__":
    os.chdir("..")
    original_tif = "data/pFIB/pristine_full.tif"
    segmented_tif = "scripts/output/segmented_stack.tif"
    overlay_segmentation(original_tif, segmented_tif, slice_index=333)
