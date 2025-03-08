# reproduce the results of the paper
import os
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
from skimage.morphology import ball
import matplotlib.pyplot as plt

# Define hyperparameters (adjust these values as needed)
sigma_initial = 1.0        # For initial Gaussian smoothing (Step 1)
gradient_threshold = 20.0  # Threshold for the forward gradient (Step 2)
low_threshold = 50         # Low grayscale threshold (Step 3)
high_threshold = 200       # High grayscale threshold (Step 4)
morph_radius = 2           # Radius for the 3D spherical structuring element (Step 5)
sigma_final = 1.0          # Sigma for final 3D Gaussian smoothing (Step 6)

def process_slice(slice_img):
    """
    Process one 2D slice:
    1. Apply Gaussian smoothing.
    2. Compute a forward difference (gradient) along the horizontal axis.
    3. Use low and high thresholds along with the gradient to assign solid/pore labels.
    """
    # Step 1: Gaussian smoothing
    smoothed = gaussian_filter(slice_img, sigma=sigma_initial)
    
    # Step 2: Compute forward gradient along horizontal axis (axis=1)
    # The 'prepend' argument ensures the output has the same shape as the input.
    gradient = np.diff(smoothed, axis=1, prepend=smoothed[:, 0:1])
    
    # Initialize binary mask (False for pore, True for solid)
    binary = np.zeros_like(smoothed, dtype=bool)
    
    # Step 3 & 4: Apply low and high grayscale thresholding
    # Directly label as solid if intensity > high_threshold, and as pore if < low_threshold.
    solid_mask = smoothed > high_threshold
    pore_mask = smoothed < low_threshold
    intermediate_mask = (~solid_mask) & (~pore_mask)
    
    binary[solid_mask] = True
    binary[pore_mask] = False
    # For intermediate intensities, use the gradient condition:
    binary[intermediate_mask] = gradient[intermediate_mask] > gradient_threshold
    
    return binary

def segment_volume(volume):
    """
    Process the entire 3D volume:
    - Process each 2D slice.
    - Stack the slices to form a 3D binary volume.
    - Apply 3D morphological cleaning.
    - Apply final 3D Gaussian smoothing and re-threshold.
    """
    segmented_slices = []
    for i in range(volume.shape[0]):
        slice_img = volume[i]
        binary_slice = process_slice(slice_img)
        segmented_slices.append(binary_slice)
    seg_volume = np.stack(segmented_slices, axis=0)
    
    # Step 5: 3D morphological cleaning using a spherical structuring element
    struct_elem = ball(morph_radius)
    seg_volume = binary_opening(seg_volume, structure=struct_elem)
    seg_volume = binary_closing(seg_volume, structure=struct_elem)
    
    # Step 6: Final 3D Gaussian smoothing and re-thresholding.
    # Gaussian smoothing on a binary volume yields values between 0 and 1.
    smoothed_volume = gaussian_filter(seg_volume.astype(np.float32), sigma=sigma_final)
    final_volume = smoothed_volume > 0.5  # Re-threshold to get a binary volume
    
    return final_volume

def main():
    os.chdir("..")
    # Load the TIFF stack (assumed to be a stack of 2D grayscale images)
    input_file = "data/pFIB/pristine_full.tif"  # Change this to your TIFF file path
    volume = tifffile.imread(input_file)
    print("Loaded volume shape:", volume.shape)
    
    # Segment the volume using the defined method
    segmented_volume = segment_volume(volume)
    
    # Save the segmented volume as a NumPy binary file for later use
    np.save("scripts/output/segmented_volume.npy", segmented_volume)
    print("Segmented volume saved as scripts/output/segmented_volume.npy")
    
    # Optionally, save the segmented volume as a TIFF stack (scaling Boolean to 0 and 255)
    tifffile.imwrite("scripts/output/segmented_stack.tif", segmented_volume.astype(np.uint8) * 255)
    print("Segmented volume saved as scripts/output/segmented_stack.tif")
    
    # Display a few example slices for visual verification
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slice_indices = [volume.shape[0] // 4, volume.shape[0] // 2, 3 * volume.shape[0] // 4]
    for ax, idx in zip(axes, slice_indices):
        ax.imshow(segmented_volume[idx], cmap='gray')
        ax.set_title(f"Segmented Slice {idx}")
        ax.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
