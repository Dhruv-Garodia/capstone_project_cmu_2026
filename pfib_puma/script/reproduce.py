# reproduce the results of the paper
# TODO: crop inside the (cropping the outside - rectangle)
# TODO: replicate the value of pososity that is consistent with the paper 0.5 / 127
# TODO: replicate the paper

import os
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, rotate
from skimage.morphology import ball
import matplotlib.pyplot as plt
from skimage import exposure

# Define hyperparameters (adjust these values as needed)
rotation_angle = 52        # Rotation angle for the slice (Step 0)
sigma_initial = 0.8        # For initial Gaussian smoothing (Step 1)
gradient_threshold = 15.0  # Threshold for the forward gradient (Step 2)
low_threshold = 40         # Low grayscale threshold (Step 3)
high_threshold = 170       # High grayscale threshold (Step 4)
morph_radius = 1           # Radius for the 3D spherical structuring element (Step 5)
sigma_final = 0.7          # Sigma for final 3D Gaussian smoothing (Step 6)

def preprocess_volume(volume, ref_slice_idx=None):
    """
    Preprocess the volume:
    - Normalize histogram of each slice to a reference slice
    
    Args:
        volume: 3D numpy array
        ref_slice_idx: Index of reference slice for histogram matching (middle slice if None)
        
    Returns:
        Preprocessed volume
    """
    print("Preprocessing volume...")
    
    # Choose middle slice as reference if not specified
    if ref_slice_idx is None:
        ref_slice_idx = volume.shape[0] // 2
    
    # Extract reference histogram
    reference = volume[ref_slice_idx]
    
    # Apply histogram matching to each slice
    matched_volume = np.zeros_like(volume)
    for i in range(volume.shape[0]):
        matched_volume[i] = exposure.match_histograms(volume[i], reference)
    
    return matched_volume

def correct_streak_artifacts(volume, angle=52):
    """
    Correct streak artifacts by:
    1. Rotating the volume to align with streak artifacts
    2. Reslicing to get planes parallel to the streak artifacts
    
    Args:
        volume: 3D numpy array with x-axis as stacked direction
        angle: Rotation angle in degrees
        
    Returns:
        Rotated and resliced volume
    """
    print(f"Correcting streak artifacts (rotating by {angle}°)...")
    
    # Since x-axis is the stacked direction, we need to rotate around the y-axis
    # This means rotation in the x-z plane (axes=(0,2))
    rotated = rotate(volume, angle, axes=(0,2), reshape=True, order=1, mode='constant')
    
    # Now we need to reslice the volume. In ImageJ terms, this would mean
    # viewing the volume from a different orthogonal plane.
    # If we're taking slices from the original stacking direction (x),
    # after rotation we should take slices from the z direction to be parallel with streaks
    resliced = np.transpose(rotated, (2, 1, 0))
    
    return resliced

def reverse_reslice_rotation(volume, angle=52):
    """
    Reverse the reslicing and rotation to return to original orientation
    
    Args:
        volume: 3D numpy array that was previously rotated and resliced
        angle: Original rotation angle in degrees
        
    Returns:
        Volume in original orientation
    """
    print("Restoring original orientation...")
    
    # First reverse the reslicing (transpose back)
    un_resliced = np.transpose(volume, (2, 1, 0))
    
    # Then rotate back by negative angle
    un_rotated = rotate(un_resliced, -angle, axes=(0,2), reshape=True, order=1, mode='constant')
    
    return un_rotated

def process_slice(slice_img):
    """
    Process a single 2D slice using steps 1-4 from the paper
    
    Args:
        slice_img: 2D numpy array
        
    Returns:
        Binary segmentation of the slice
    """
    # Step 1: Gaussian smoothing
    smoothed = gaussian_filter(slice_img, sigma=sigma_initial)
    
    # Step 2: Compute forward gradient along horizontal axis (axis=1)
    gradient = np.zeros_like(smoothed)
    gradient[:, 1:] = np.diff(smoothed, axis=1)
    
    # Initialize binary mask (False for pore, True for solid)
    binary = np.zeros_like(smoothed, dtype=bool)
    
    # Step 3 & 4: Apply low and high grayscale thresholding
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
    Process the entire 3D volume following the paper's methodology:
    1. Preprocess (histogram normalization)
    2. Correct streak artifacts (rotate and reslice)
    3. Process each 2D slice (steps 1-4)
    4. Apply 3D morphological operations (step 5)
    5. Apply 3D Gaussian smoothing (step 6)
    6. Restore original orientation
    
    Args:
        volume: Original 3D numpy array
        
    Returns:
        Segmented binary 3D volume in original orientation
    """
    # Step 0: Preprocess
    preprocessed = preprocess_volume(volume)
    
    # Step 0: Correct streak artifacts (rotate and reslice)
    artifact_corrected = correct_streak_artifacts(preprocessed, angle=rotation_angle)
    
    # Steps 1-4: Process each 2D slice
    print(f"Processing {artifact_corrected.shape[0]} slices...")
    segmented_slices = []
    for i in range(artifact_corrected.shape[0]):
        if i % 20 == 0:
            print(f"  - Processing slice {i}/{artifact_corrected.shape[0]}")
        slice_img = artifact_corrected[i]
        binary_slice = process_slice(slice_img)
        segmented_slices.append(binary_slice)
    seg_volume = np.stack(segmented_slices, axis=0)
    
    # Step 5: 3D morphological cleaning using a spherical structuring element
    print("Applying 3D morphological operations...")
    struct_elem = ball(morph_radius)
    seg_volume = binary_opening(seg_volume, structure=struct_elem)
    seg_volume = binary_closing(seg_volume, structure=struct_elem)
    
    # Step 6: Final 3D Gaussian smoothing and re-thresholding
    print("Applying final 3D Gaussian smoothing...")
    smoothed_volume = gaussian_filter(seg_volume.astype(np.float32), sigma=sigma_final)
    final_volume = smoothed_volume > 0.5  # Re-threshold to get a binary volume
    
    # Restore original orientation
    restored_volume = reverse_reslice_rotation(final_volume, angle=rotation_angle)
    
    return restored_volume

def main():
    input_file = "data/pFIB/pristine_full.tif"
    volume = tifffile.imread(input_file)
    print("Loaded volume shape:", volume.shape)
    
    segmented_volume = segment_volume(volume)
    # To get exactly 665 slices
    extracted_volume = segmented_volume[967:967+665]
    
    np.save("scripts/output/segmented_volume.npy", extracted_volume)
    print("Extracted volume saved as scripts/output/segmented_volume.npy")
    
    tifffile.imwrite("scripts/output/segmented_stack.tif", extracted_volume.astype(np.uint8) * 255)
    print("Extracted volume saved as scripts/output/segmented_stack.tif")
    
if __name__ == "__main__":
    main()
