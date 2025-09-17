import os
import sys
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt

def analyze_porosity(stack, label="Stack"):
    """
    Calculate porosity metrics for a complete image stack (original approach).
    
    Args:
        stack: 3D numpy array with image data
        label: Label for the analysis output
        
    Returns:
        Dictionary with porosity metrics
    """
    print(f"{label} shape:", stack.shape)
    
    # Calculate overall porosity
    porosity = np.sum(stack) / stack.size * 100  # Convert to percentage
    print(f"Overall porosity: {porosity:.4f}%")
    
    # Calculate slice-by-slice porosity
    slice_porosities = [np.sum(stack[i]) / stack[i].size * 100 for i in range(stack.shape[0])]
    
    print(f"Min slice porosity: {min(slice_porosities):.4f}%")
    print(f"Max slice porosity: {max(slice_porosities):.4f}%")
    print(f"Mean slice porosity: {np.mean(slice_porosities):.4f}%")
    print(f"Std dev of slice porosity: {np.std(slice_porosities):.4f}%")
    
    # Save statistics to file
    with open("scripts/output/porosity_stats.txt", "w") as f:
        f.write(f"=== {label} Porosity Analysis ===\n\n")
        f.write(f"Shape: {stack.shape}\n")
        f.write(f"Overall porosity: {porosity:.4f}%\n\n")
        f.write(f"Slice porosity - \n")
        f.write(f"  Min: {min(slice_porosities):.4f}%\n")
        f.write(f"  Max: {max(slice_porosities):.4f}%\n")
        f.write(f"  Mean: {np.mean(slice_porosities):.4f}%\n")
        f.write(f"  StdDev: {np.std(slice_porosities):.4f}%\n")
    
    print(f"Porosity statistics saved to scripts/output/porosity_stats.txt")
    
    # Create porosity profile plot
    plt.figure(figsize=(10, 6))
    plt.plot(slice_porosities, 'b-', alpha=0.7)
    plt.xlabel('Slice Index')
    plt.ylabel('Porosity (%)')
    plt.title('Slice-by-Slice Porosity')
    plt.grid(True, alpha=0.3)
    plt.savefig("scripts/output/porosity_profile.png", dpi=300)
    print("Porosity profile saved to scripts/output/porosity_profile.png")
    plt.close()
    
    return {
        'shape': stack.shape,
        'porosity': porosity,
        'min_porosity': min(slice_porosities),
        'max_porosity': max(slice_porosities),
        'mean_porosity': np.mean(slice_porosities),
        'std_porosity': np.std(slice_porosities)
    }

def analyze_region_porosity(stack, highlighted_regions, label="Region", pore_is_white=True, threshold_factor=0.8):
    """
    Calculate porosity metrics for a specific region of interest in an image stack.
    
    Args:
        stack: 3D numpy array with image data
        highlighted_regions: 3D boolean mask or grayscale image with highlighted regions
        label: Label for the analysis output
        pore_is_white: If True, higher values (white) are pores; if False, lower values (black) are pores
        threshold_factor: Factor to multiply mean intensity by for thresholding grayscale images
        
    Returns:
        Dictionary with porosity metrics
    """
    print(f"Stack shape: {stack.shape}")
    print(f"Highlighted regions shape: {highlighted_regions.shape}")
    
    # Convert highlighted regions to boolean mask if needed
    if highlighted_regions.dtype != bool:
        # If it's a grayscale image, areas with max value are highlighted
        max_val = np.max(highlighted_regions)
        if max_val > 0:
            highlighted_mask = highlighted_regions == max_val
        else:
            # Fallback if there are no highlighted regions
            highlighted_mask = np.zeros_like(highlighted_regions, dtype=bool)
    else:
        highlighted_mask = highlighted_regions
    
    # Calculate total number of voxels in highlighted regions
    total_highlighted_voxels = np.sum(highlighted_mask)
    
    # If there are no highlighted voxels, return early
    if total_highlighted_voxels == 0:
        print("Error: No highlighted regions found in the mask")
        return {
            'shape': stack.shape,
            'porosity': 0,
            'min_porosity': 0,
            'max_porosity': 0,
            'mean_porosity': 0,
            'std_porosity': 0,
            'highlighted_voxels': 0
        }
    
    # Calculate porosity within highlighted regions only
    # Determine what's considered a pore based on pore_is_white parameter
    
    # For binary images, sum directly
    if stack.dtype == bool:
        if pore_is_white:
            # White/True is pore
            pore_mask = stack & highlighted_mask
        else:
            # Black/False is pore
            pore_mask = (~stack) & highlighted_mask
    # For grayscale, threshold at mean value
    else:
        threshold = np.mean(stack) * threshold_factor
        if pore_is_white:
            # Higher values (> threshold) are pores
            pore_mask = (stack > threshold) & highlighted_mask
        else:
            # Lower values (< threshold) are pores
            pore_mask = (stack < threshold) & highlighted_mask
    
    pore_voxels = np.sum(pore_mask)
    
    # Calculate overall porosity in the region
    porosity = (pore_voxels / total_highlighted_voxels) * 100  # Convert to percentage
    print(f"Total highlighted voxels: {total_highlighted_voxels}")
    print(f"Pore voxels within highlighted regions: {pore_voxels}")
    print(f"Overall region porosity: {porosity:.4f}%")
    
    # Calculate slice-by-slice porosity in highlighted regions
    slice_porosities = []
    slice_highlighted_voxels = []
    
    for i in range(stack.shape[0]):
        # Count highlighted voxels in this slice
        slice_total = np.sum(highlighted_mask[i])
        slice_highlighted_voxels.append(slice_total)
        
        # Skip slices with no highlighted areas
        if slice_total == 0:
            continue
        
        # Count pore voxels in highlighted area for this slice
        slice_pores = np.sum(pore_mask[i])
        
        # Calculate slice porosity
        slice_porosity = (slice_pores / slice_total) * 100
        slice_porosities.append(slice_porosity)
    
    # Handle case with no valid slice porosities
    if not slice_porosities:
        print("Warning: No slices with highlighted regions found")
        min_porosity, max_porosity, mean_porosity, std_porosity = 0, 0, 0, 0
    else:
        min_porosity = min(slice_porosities)
        max_porosity = max(slice_porosities)
        mean_porosity = np.mean(slice_porosities)
        std_porosity = np.std(slice_porosities)
    
    print(f"Min slice porosity: {min_porosity:.4f}%")
    print(f"Max slice porosity: {max_porosity:.4f}%")
    print(f"Mean slice porosity: {mean_porosity:.4f}%")
    print(f"Std dev of slice porosity: {std_porosity:.4f}%")
    
    # Save statistics to file
    with open("scripts/output/region_porosity_stats.txt", "w") as f:
        f.write(f"=== {label} Porosity Analysis ===\n\n")
        f.write(f"Stack shape: {stack.shape}\n")
        f.write(f"Total highlighted voxels: {total_highlighted_voxels}\n")
        f.write(f"Pore voxels within highlighted regions: {pore_voxels}\n")
        f.write(f"Overall region porosity: {porosity:.4f}%\n\n")
        f.write(f"Slice porosity within highlighted regions - \n")
        f.write(f"  Min: {min_porosity:.4f}%\n")
        f.write(f"  Max: {max_porosity:.4f}%\n")
        f.write(f"  Mean: {mean_porosity:.4f}%\n")
        f.write(f"  StdDev: {std_porosity:.4f}%\n")
    
    print(f"Region porosity statistics saved to scripts/output/region_porosity_stats.txt")
    
    # Create porosity profile plot
    create_porosity_profile(slice_porosities, slice_highlighted_voxels, stack.shape[0])
    
    return {
        'shape': stack.shape,
        'porosity': porosity,
        'min_porosity': min_porosity,
        'max_porosity': max_porosity,
        'mean_porosity': mean_porosity,
        'std_porosity': std_porosity,
        'highlighted_voxels': total_highlighted_voxels,
        'pore_voxels': pore_voxels
    }

def create_porosity_profile(slice_porosities, slice_highlighted_voxels, total_slices):
    """Create a plot showing the porosity profile and highlighted region size through the stack"""
    # Create indices for slices with data
    valid_indices = [i for i, count in enumerate(slice_highlighted_voxels) if count > 0]
    
    if not valid_indices:
        print("Cannot create porosity profile: no slices with highlighted regions")
        return
    
    # Create figure with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot porosity profile
    porosity_values = [slice_porosities[valid_indices.index(i)] for i in valid_indices]
    ax1.plot(valid_indices, porosity_values, 'b-', marker='o', alpha=0.7)
    ax1.set_ylabel('Porosity (%)')
    ax1.set_title('Porosity Profile within Highlighted Regions')
    ax1.grid(True, alpha=0.3)
    
    # Plot highlighted voxel count per slice
    nonzero_counts = [slice_highlighted_voxels[i] for i in valid_indices]
    ax2.bar(valid_indices, nonzero_counts, alpha=0.5, color='green')
    ax2.set_xlabel('Slice Index')
    ax2.set_ylabel('Highlighted Voxels')
    ax2.set_xlim(0, total_slices - 1)
    
    plt.tight_layout()
    plt.savefig("scripts/output/region_porosity_profile.png", dpi=300)
    print("Porosity profile saved to scripts/output/region_porosity_profile.png")
    plt.close()

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
    parser = argparse.ArgumentParser(description='Calculate porosity of image stacks')
    parser.add_argument('stack_file', help='Path to image stack file (TIFF or NPY)')
    parser.add_argument('highlight_file', nargs='?', default=None, 
                       help='Path to highlighted regions file (optional)')
    parser.add_argument('--pores-are-black', '-b', action='store_true', 
                        help='Treat black/dark areas as pores (default: white/bright areas are pores)')
    parser.add_argument('--label', '-l', type=str, default=None, 
                        help='Custom label for the analysis')
    parser.add_argument('--threshold', '-t', type=float, default=0.8, 
                        help='Threshold factor for porosity detection (0-1, default: 0.8)')
    parser.add_argument('--no-chdir', action='store_true', 
                        help='Do not change to project root directory')
    
    args = parser.parse_args()
    
    # Change to project root directory if not disabled
    if not args.no_chdir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        os.chdir(project_root)
        print(f"Working directory: {os.getcwd()}")
    
    # Create output directory if it doesn't exist
    os.makedirs("scripts/output", exist_ok=True)
    
    # Get default label from filename if not provided
    if args.label is None:
        args.label = os.path.basename(args.stack_file).split('.')[0].capitalize()
    
    # Load stack file
    try:
        print(f"Loading image stack from {args.stack_file}...")
        stack = load_stack(args.stack_file)
        print(f"Loaded stack with shape: {stack.shape}, type: {stack.dtype}")
        
        # Determine which analysis to run based on whether highlight file is provided
        if args.highlight_file is None:
            # Run original porosity analysis on the entire stack
            print("No highlight file provided, analyzing entire stack...")
            analyze_porosity(stack, label=args.label)
        else:
            # Load highlighted regions and run region-specific analysis
            print(f"Loading highlighted regions from {args.highlight_file}...")
            highlighted = load_stack(args.highlight_file)
            print(f"Loaded highlighted regions with shape: {highlighted.shape}, type: {highlighted.dtype}")
            
            # Verify both stacks have the same shape
            if stack.shape != highlighted.shape:
                print("Warning: Stack and highlighted regions have different shapes")
                print(f"Stack: {stack.shape}, Highlighted: {highlighted.shape}")
                
                # Try to crop to the same shape
                min_z = min(stack.shape[0], highlighted.shape[0])
                min_y = min(stack.shape[1], highlighted.shape[1])
                min_x = min(stack.shape[2], highlighted.shape[2])
                
                stack = stack[:min_z, :min_y, :min_x]
                highlighted = highlighted[:min_z, :min_y, :min_x]
                print(f"Cropped to common shape: {stack.shape}")
            
            # Run analysis with specified pore convention
            pore_is_white = not args.pores_are_black
            print(f"Analyzing with {'black' if args.pores_are_black else 'white'} as pores...")
            
            analyze_region_porosity(
                stack, 
                highlighted, 
                label=args.label,
                pore_is_white=pore_is_white,
                threshold_factor=args.threshold
            )
        
    except Exception as e:
        print(f"Error analyzing stack: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()