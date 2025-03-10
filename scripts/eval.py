import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile
def validate_segmentation(original, segmented):
    """Basic validation of segmentation results"""
    print("Original shape:", original.shape)
    print("Segmented shape:", segmented.shape)
    
    # area of the solid part / area of the total part
    porosity = np.sum(segmented) / segmented.size
    print(f"Overall porosity for segmented volume: {porosity:.3f}")
    
    porosity_original = np.sum(original) / original.size
    print(f"Overall porosity for original volume: {porosity_original:.3f}")
    
    # Check for consistency across slices
    slice_porosities = [np.sum(segmented[i]) / segmented[i].size 
                       for i in range(segmented.shape[0])]
    print(f"Min slice porosity: {min(slice_porosities):.3f}")
    print(f"Max slice porosity: {max(slice_porosities):.3f}")
    print(f"Std dev of slice porosity: {np.std(slice_porosities):.3f}")
    
    slice_porosities_original = [np.sum(original[i]) / original[i].size 
                                 for i in range(original.shape[0])]
    print(f"Min slice porosity for original volume: {min(slice_porosities_original):.3f}")
    print(f"Max slice porosity for original volume: {max(slice_porosities_original):.3f}")
    print(f"Std dev of slice porosity for original volume: {np.std(slice_porosities_original):.3f}")
    
    # Visualize a sample of slices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    indices = np.linspace(0, min(original.shape[0], segmented.shape[0])-1, 3).astype(int)
    
    for i, idx in enumerate(indices):
        # Original
        if idx < original.shape[0]:
            axes[0, i].imshow(original[idx], cmap='gray')
            axes[0, i].set_title(f"Original #{idx}")
        axes[0, i].axis('off')
        
        # Segmented
        if idx < segmented.shape[0]:
            axes[1, i].imshow(segmented[idx], cmap='gray')
            axes[1, i].set_title(f"Segmented #{idx}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    os.chdir("..")
    original = tifffile.imread("data/pFIB/pristine_full.tif")
    segmented = tifffile.imread("scripts/output/segmented_stack.tif")
    validate_segmentation(original, segmented)