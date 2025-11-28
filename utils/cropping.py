import os
import sys
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage.measure import label, regionprops
from skimage.segmentation import mark_boundaries

def manually_select_regions(image_stack):
    """
    Allow user to manually select regions of interest in multiple slices.
    
    Args:
        image_stack: 3D numpy array with shape (z, y, x)
        
    Returns:
        List of ROI coordinates for key slices and labels for all slices
    """
    # Define key slices to mark (non-zero index for first one, 1/4, middle, 3/4, end)
    total_slices = image_stack.shape[0]
    key_indices = [
        min(1, total_slices - 1),  # Second slice instead of first (which might be black)
        total_slices // 4,         # Quarter mark
        total_slices // 2,         # Middle
        3 * total_slices // 4,     # Three-quarter mark
        total_slices - 1           # Last slice
    ]
    
    # Create a mask volume with the same shape as the image stack
    mask_volume = np.zeros_like(image_stack, dtype=bool)
    
    # Dictionary to store region coordinates for key slices
    key_regions = {}
    
    # Let user select regions for key slices
    for idx in key_indices:
        print(f"\nSelecting region for slice {idx} of {total_slices-1}")
        
        fig, ax = plt.figure(figsize=(10, 8)), plt.subplot(111)
        ax.imshow(image_stack[idx], cmap='gray')
        plt.title(f'Select region of interest in slice {idx}')
        
        # Initialize region coordinates
        region_coords = {}
        
        def line_select_callback(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            region_coords['x_min'] = min(x1, x2)
            region_coords['y_min'] = min(y1, y2)
            region_coords['x_max'] = max(x1, x2)
            region_coords['y_max'] = max(y1, y2)
            
        rs = RectangleSelector(
            ax, line_select_callback,
            useblit=True,
            button=[1],  # Left mouse button only
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        plt.tight_layout()
        plt.show()
        
        if not region_coords:
            print(f"No region selected for slice {idx}, skipping...")
            continue
            
        # Store region coordinates
        key_regions[idx] = (
            region_coords['x_min'],
            region_coords['y_min'],
            region_coords['x_max'],
            region_coords['y_max']
        )
        
        # Create mask for this slice
        x_min, y_min, x_max, y_max = key_regions[idx]
        mask = np.zeros_like(image_stack[idx], dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        mask_volume[idx] = mask
        
        print(f"Selected region for slice {idx}: x=[{x_min}-{x_max}], y=[{y_min}-{y_max}]")
    
    # Interpolate regions for all slices between key slices
    for i in range(total_slices):
        # Skip key slices that were already processed
        if i in key_regions:
            continue
        
        # Find the closest key slices before and after this slice
        prev_keys = [k for k in key_indices if k <= i and k in key_regions]
        next_keys = [k for k in key_indices if k >= i and k in key_regions]
        
        if not prev_keys or not next_keys:
            # Skip slices outside the range of selected key slices
            continue
        
        prev_key = max(prev_keys)
        next_key = min(next_keys)
        
        # If between two key slices, interpolate
        if prev_key != next_key:
            # Calculate how far we are between the two key slices (0 to 1)
            weight = (i - prev_key) / (next_key - prev_key)
            
            # Get region coordinates for the key slices
            prev_region = key_regions[prev_key]
            next_region = key_regions[next_key]
            
            # Interpolate each coordinate
            interp_region = (
                int(prev_region[0] * (1 - weight) + next_region[0] * weight),
                int(prev_region[1] * (1 - weight) + next_region[1] * weight),
                int(prev_region[2] * (1 - weight) + next_region[2] * weight),
                int(prev_region[3] * (1 - weight) + next_region[3] * weight)
            )
            
            # Create mask for this slice
            x_min, y_min, x_max, y_max = interp_region
            mask = np.zeros_like(image_stack[i], dtype=bool)
            mask[y_min:y_max, x_min:x_max] = True
            mask_volume[i] = mask
    
    return key_regions, mask_volume

def track_shifting_region(image_stack, initial_region, vertical_shift_per_slice=0.1):
    """
    Track a region across the image stack with a gradual vertical shift.
    
    Args:
        image_stack: 3D numpy array with shape (z, y, x)
        initial_region: Tuple of (x_min, y_min, x_max, y_max) for first slice
        vertical_shift_per_slice: How much to shift up (negative) or down (positive) per slice
        
    Returns:
        3D boolean mask of same shape as image_stack
    """
    x_min, y_min, x_max, y_max = initial_region
    height = y_max - y_min
    width = x_max - x_min
    
    # Create output mask
    mask_volume = np.zeros_like(image_stack, dtype=bool)
    
    # Calculate the vertical shift for each slice
    for i in range(image_stack.shape[0]):
        # Calculate the shift for this slice
        shift = int(i * vertical_shift_per_slice)
        
        # Calculate new y coordinates ensuring they stay within image bounds
        new_y_min = max(0, y_min - shift)
        new_y_max = min(image_stack.shape[1], new_y_min + height)
        
        # If we hit the image bound, adjust to maintain the same height
        if new_y_max == image_stack.shape[1]:
            new_y_min = new_y_max - height
        
        # Create mask for this slice
        mask = np.zeros_like(image_stack[i], dtype=bool)
        mask[new_y_min:new_y_max, x_min:x_max] = True
        mask_volume[i] = mask
    
    return mask_volume

def auto_detect_regions(image_stack, padding=10, margin_top=0, margin_bottom=0, margin_left=0, margin_right=0):
    """
    Automatically detect regions of interest based on grayscale intensity analysis.
    
    Args:
        image_stack: 3D numpy array with shape (z, y, x)
        padding: Additional padding around the detected region
        margin_top, margin_bottom, margin_left, margin_right: Margins to respect
        
    Returns:
        3D boolean mask of same shape as image_stack
    """
    print("Detecting regions of interest...")
    
    # Initialize output mask
    mask_volume = np.zeros_like(image_stack, dtype=bool)
    
    # Sample slices from beginning, middle, and end
    slice_indices = [0, image_stack.shape[0] // 2, image_stack.shape[0] - 1]
    
    # Process each slice to find the region bounds
    all_regions = []
    for idx in slice_indices:
        slice_img = image_stack[idx]
        
        # Apply threshold to separate foreground from background
        if slice_img.dtype == np.bool_:
            binary_slice = slice_img
        else:
            # For grayscale, use a simple threshold
            threshold = np.mean(slice_img) * 0.8  # Adjust as needed
            binary_slice = slice_img > threshold
        
        # Find rows and columns with content
        row_sums = np.sum(binary_slice, axis=1)
        col_sums = np.sum(binary_slice, axis=0)
        
        content_rows = np.where(row_sums > (binary_slice.shape[1] * 0.05))[0]
        content_cols = np.where(col_sums > (binary_slice.shape[0] * 0.05))[0]
        
        if len(content_rows) == 0 or len(content_cols) == 0:
            continue
        
        # Get region bounds with padding and margin constraints
        y_min = max(margin_top, content_rows[0] - padding)
        y_max = min(image_stack.shape[1] - margin_bottom, content_rows[-1] + padding)
        x_min = max(margin_left, content_cols[0] - padding)
        x_max = min(image_stack.shape[2] - margin_right, content_cols[-1] + padding)
        
        all_regions.append((x_min, y_min, x_max, y_max))
    
    # If no regions found, use default
    if not all_regions:
        print("No regions detected, using default")
        x_min = margin_left
        y_min = margin_top
        x_max = image_stack.shape[2] - margin_right
        y_max = image_stack.shape[1] - margin_bottom
        all_regions.append((x_min, y_min, x_max, y_max))
    
    # Get the overall bounding box that contains all detected regions
    x_min = min(region[0] for region in all_regions)
    y_min = min(region[1] for region in all_regions)
    x_max = max(region[2] for region in all_regions)
    y_max = max(region[3] for region in all_regions)
    
    # Create initial region
    initial_region = (x_min, y_min, x_max, y_max)
    
    # Now track this region across slices to account for vertical movement
    return initial_region, track_shifting_region(image_stack, initial_region, -0.15)

def create_overlay_images(image_stack, mask_volume, output_dir="scripts/output/overlays"):
    """
    Create overlay images showing the labeled regions on original slices.
    
    Args:
        image_stack: 3D numpy array with original image data
        mask_volume: 3D boolean array with labeled regions
        output_dir: Directory to save overlay images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving overlay images to {output_dir}...")
    
    # Select slices to save (we'll save a subset to avoid too many files)
    num_slices = image_stack.shape[0]
    save_indices = np.linspace(0, num_slices-1, min(20, num_slices)).astype(int)
    
    # Create and save overlay images
    for idx in save_indices:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Show original image
        ax.imshow(image_stack[idx], cmap='gray')
        
        # Overlay mask boundary in red
        mask = mask_volume[idx]
        if np.any(mask):
            overlay = mark_boundaries(np.zeros_like(image_stack[idx]), mask, color=(1, 0, 0))
            ax.imshow(overlay, alpha=0.7)
        
        ax.set_title(f"Slice {idx}")
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"overlay_slice_{idx:04d}.png"), dpi=150)
        plt.close()
    
    # Also create a composite image showing several slices
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    display_indices = np.linspace(0, num_slices-1, 10).astype(int)
    
    for i, idx in enumerate(display_indices):
        axes[i].imshow(image_stack[idx], cmap='gray')
        
        mask = mask_volume[idx]
        if np.any(mask):
            overlay = mark_boundaries(np.zeros_like(image_stack[idx]), mask, color=(1, 0, 0))
            axes[i].imshow(overlay, alpha=0.7)
        
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "composite_overlay.png"), dpi=200)
    plt.close()
    
    print(f"Saved {len(save_indices)} overlay images and a composite view.")

def save_labeled_volume(image_stack, mask_volume, output_dir="scripts/output"):
    """
    Save the original volume and labeled mask as TIFF and NumPy files.
    
    Args:
        image_stack: 3D numpy array with original image data
        mask_volume: 3D boolean array with labeled regions
        output_dir: Directory to save files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mask volume as TIFF
    tiff_path = os.path.join(output_dir, "labeled_regions.tif")
    tifffile.imwrite(tiff_path, mask_volume.astype(np.uint8) * 255)
    print(f"Labeled regions saved to {tiff_path}")
    
    # Save mask volume as NumPy
    npy_path = os.path.join(output_dir, "labeled_regions.npy")
    np.save(npy_path, mask_volume)
    print(f"Labeled regions saved to {npy_path}")
    
    # Create a version of the original with masked areas highlighted
    highlighted = image_stack.copy()
    # Scale the original image to 0-200 range to leave room for highlighting
    if highlighted.dtype != np.bool_:
        highlighted = (highlighted.astype(float) * (200 / 255)).astype(np.uint8)
    # Set the masked regions to 255 (white/bright)
    highlighted[mask_volume] = 255
    
    # Save highlighted volume as TIFF
    highlighted_path = os.path.join(output_dir, "highlighted_stack.tif")
    tifffile.imwrite(highlighted_path, highlighted)
    print(f"Highlighted stack saved to {highlighted_path}")

def load_stack(path):
    """Load an image stack with smart path handling"""
    # Check if the file exists directly
    if os.path.exists(path):
        if path.endswith('.npy'):
            return np.load(path)
        else:
            return tifffile.imread(path)
            
    # Extract just the filename
    filename = os.path.basename(path)
    
    # Common locations to check
    common_locations = [
        os.path.join("scripts/output", filename),
        os.path.join("output", filename),
        os.path.join("data/pFIB", filename)
    ]
    
    # Try each location
    for location in common_locations:
        if os.path.exists(location):
            if location != path:
                print(f"Using {location}")
            if location.endswith('.npy'):
                return np.load(location)
            else:
                return tifffile.imread(location)
    
    # If we get here, file wasn't found
    raise FileNotFoundError(f"Could not find {filename} in any common directory")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Label regions of interest in an image stack')
    parser.add_argument('input_file', nargs='?', type=str, default="scripts/output/segmented_stack.tif", 
                        help='Path to input image stack file')
    parser.add_argument('--mode', '-m', type=str, choices=['manual', 'auto', 'preset'], 
                        help='Labeling mode: manual, auto, or preset')
    parser.add_argument('--output-dir', type=str, default="scripts/output", 
                        help='Directory to save output files')
    parser.add_argument('--shift-rate', '-s', type=float, default=-0.15,
                        help='Vertical shift rate in pixels per slice (negative for upward)')
    parser.add_argument('--margin-top', type=int, default=180,
                        help='Top margin in pixels')
    parser.add_argument('--margin-bottom', type=int, default=150,
                        help='Bottom margin in pixels')
    parser.add_argument('--margin-left', type=int, default=20,
                        help='Left margin in pixels')
    parser.add_argument('--margin-right', type=int, default=20,
                        help='Right margin in pixels')
    parser.add_argument('--skip-overlays', action='store_true', 
                        help='Skip creation of overlay images')
    
    args = parser.parse_args()
    
    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Load the image stack
    try:
        print(f"Loading {args.input_file}...")
        image_stack = load_stack(args.input_file)
        print(f"Loaded stack with shape: {image_stack.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # If mode not specified, ask the user interactively
    label_mode = args.mode
    if label_mode is None:
        while True:
            label_mode = input("Choose labeling method (m for manual, a for auto, p for preset): ").strip().lower()
            if label_mode in ['m', 'a', 'p']:
                break
            print("Invalid input. Please enter 'm', 'a', or 'p'.")
        
        # Convert short form to full form
        if label_mode == 'm':
            label_mode = 'manual'
        elif label_mode == 'a':
            label_mode = 'auto'
        else:
            label_mode = 'preset'
    
    try:
        # Manual labeling with tracking
        if label_mode == 'manual':
            print("\nYou'll select regions of interest in key slices, and we'll interpolate between them.")
            key_regions, mask_volume = manually_select_regions(image_stack)
            
        # Auto detection
        elif label_mode == 'auto':
            # Automatically detect regions and track them across slices
            initial_region, mask_volume = auto_detect_regions(
                image_stack, 
                padding=10,
                margin_top=args.margin_top,
                margin_bottom=args.margin_bottom,
                margin_left=args.margin_left,
                margin_right=args.margin_right
            )
            
        # Preset with shifting
        else:  # preset
            # Calculate initial region coordinates
            x_min = args.margin_left
            y_min = args.margin_top
            x_max = image_stack.shape[2] - args.margin_right
            y_max = image_stack.shape[1] - args.margin_bottom
            
            initial_region = (x_min, y_min, x_max, y_max)
            
            # Track the region across slices
            mask_volume = track_shifting_region(image_stack, initial_region, args.shift_rate)
        
        # Save the labeled regions and highlighted stack
        save_labeled_volume(image_stack, mask_volume, args.output_dir)
        
        # Create overlay images unless skipped
        if not args.skip_overlays:
            overlay_dir = os.path.join(args.output_dir, "overlays")
            create_overlay_images(image_stack, mask_volume, overlay_dir)
        
    except Exception as e:
        print(f"Error during region labeling: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()