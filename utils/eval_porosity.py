import os
import sys
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image


def analyze_porosity(stack, label="Stack", pores_are_black=True, threshold=None):
    """
    Compute porosity (0–100%) for an entire 3D volume, and plot slice-by-slice porosity.

    Supports:
      - Boolean array: True/1 = pore or solid depending on pores_are_black
      - uint8 images: 0/255 or general grayscale (threshold applied)
    """
    print(f"{label} shape:", stack.shape, "dtype:", stack.dtype)

    # --- 1. Convert to a 0/1 pore mask ---
    if stack.dtype == bool:
        # Boolean array: True = 1, False = 0
        if pores_are_black:
            # If black means pore → assume False(0) is black, so pores = ~stack
            pore_mask = ~stack
        else:
            pore_mask = stack
    else:
        # Non-boolean → convert to float first
        arr = stack.astype(np.float32)

        # If values are in [0,1], use 0.5 as threshold
        if arr.max() <= 1.0:
            solid_mask = arr > 0.5
        else:
            # Common 0–255 grayscale: default threshold = 127 unless overridden
            if threshold is None:
                thr = 127.0
            else:
                thr = threshold
            solid_mask = arr > thr

        if pores_are_black:
            # If black means pore → solid = white → pore = not solid
            pore_mask = ~solid_mask
        else:
            pore_mask = solid_mask

    # pore_mask is bool; True = pore
    pore_mask = pore_mask.astype(bool)

    # --- 2. Compute overall porosity ---
    porosity = pore_mask.mean() * 100.0
    print(f"Overall porosity: {porosity:.4f}%")

    # --- 3. Compute each slice's porosity ---
    slice_porosities = []
    Z = pore_mask.shape[0]
    for z in range(Z):
        slice_porosities.append(pore_mask[z].mean() * 100.0)

    print(f"Min slice porosity: {min(slice_porosities):.4f}%")
    print(f"Max slice porosity: {max(slice_porosities):.4f}%")
    print(f"Mean slice porosity: {np.mean(slice_porosities):.4f}%")
    print(f"Std dev of slice porosity: {np.std(slice_porosities):.4f}%")

    os.makedirs("scripts/output", exist_ok=True)
    with open("scripts/output/porosity_stats.txt", "w") as f:
        f.write(f"=== {label} Porosity Analysis ===\n\n")
        f.write(f"Shape: {stack.shape}, dtype: {stack.dtype}\n")
        f.write(f"Overall porosity: {porosity:.4f}%\n\n")
        f.write("Slice porosity:\n")
        f.write(f"  Min: {min(slice_porosities):.4f}%\n")
        f.write(f"  Max: {max(slice_porosities):.4f}%\n")
        f.write(f"  Mean: {np.mean(slice_porosities):.4f}%\n")
        f.write(f"  StdDev: {np.std(slice_porosities):.4f}%\n")

    # --- 4. Create porosity plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(slice_porosities, 'b-', alpha=0.7)
    plt.xlabel('Slice Index')
    plt.ylabel('Porosity (%)')
    plt.title(f'Slice-by-Slice Porosity ({label})')
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
        'std_porosity': np.std(slice_porosities),
    }


def analyze_region_porosity(
    stack,
    highlighted_regions,
    label="Region",
    pore_is_white=True,
    threshold_factor=0.8,
):
    """
    Calculate porosity metrics for a region of interest in a stack.
    """
    print(f"Stack shape: {stack.shape}")
    print(f"Highlighted regions shape: {highlighted_regions.shape}")

    # Convert highlighted region to boolean mask if needed
    if highlighted_regions.dtype != bool:
        max_val = np.max(highlighted_regions)
        # Use the maximum value as the "highlighted" indicator
        if max_val > 0:
            highlighted_mask = highlighted_regions == max_val
        else:
            highlighted_mask = np.zeros_like(highlighted_regions, dtype=bool)
    else:
        highlighted_mask = highlighted_regions

    total_highlighted_voxels = np.sum(highlighted_mask)

    if total_highlighted_voxels == 0:
        print("Error: No highlighted regions found in mask")
        return {
            "shape": stack.shape,
            "porosity": 0,
            "min_porosity": 0,
            "max_porosity": 0,
            "mean_porosity": 0,
            "std_porosity": 0,
            "highlighted_voxels": 0,
        }

    # Determine pore voxels depending on pore_is_white & binary/grayscale stack
    if stack.dtype == bool:
        if pore_is_white:
            pore_mask = stack & highlighted_mask
        else:
            pore_mask = (~stack) & highlighted_mask
    else:
        # Threshold based on global mean * factor
        threshold = np.mean(stack) * threshold_factor
        if pore_is_white:
            pore_mask = (stack > threshold) & highlighted_mask
        else:
            pore_mask = (stack < threshold) & highlighted_mask

    pore_voxels = np.sum(pore_mask)
    porosity = pore_voxels / total_highlighted_voxels * 100
    print(f"Total highlighted voxels: {total_highlighted_voxels}")
    print(f"Pore voxels in highlighted regions: {pore_voxels}")
    print(f"Overall region porosity: {porosity:.4f}%")

    slice_porosities = []
    slice_highlighted_voxels = []

    for i in range(stack.shape[0]):
        slice_total = np.sum(highlighted_mask[i])
        slice_highlighted_voxels.append(slice_total)

        if slice_total == 0:
            continue

        slice_pores = np.sum(pore_mask[i])
        slice_porosities.append(slice_pores / slice_total * 100)

    if not slice_porosities:
        print("Warning: No slices with highlighted regions found")
        min_porosity = max_porosity = mean_porosity = std_porosity = 0
    else:
        min_porosity = min(slice_porosities)
        max_porosity = max(slice_porosities)
        mean_porosity = np.mean(slice_porosities)
        std_porosity = np.std(slice_porosities)

    print(f"Min slice porosity: {min_porosity:.4f}%")
    print(f"Max slice porosity: {max_porosity:.4f}%")
    print(f"Mean slice porosity: {mean_porosity:.4f}%")
    print(f"Std dev of slice porosity: {std_porosity:.4f}%")

    os.makedirs("scripts/output", exist_ok=True)
    with open("scripts/output/region_porosity_stats.txt", "w") as f:
        f.write(f"=== {label} Region Porosity Analysis ===\n\n")
        f.write(f"Stack shape: {stack.shape}\n")
        f.write(f"Total highlighted voxels: {total_highlighted_voxels}\n")
        f.write(f"Pore voxels within highlighted regions: {pore_voxels}\n")
        f.write(f"Overall region porosity: {porosity:.4f}%\n\n")
        f.write("Slice porosity within highlighted regions:\n")
        f.write(f"  Min: {min_porosity:.4f}%\n")
        f.write(f"  Max: {max_porosity:.4f}%\n")
        f.write(f"  Mean: {mean_porosity:.4f}%\n")
        f.write(f"  StdDev: {std_porosity:.4f}%\n")

    print("Region porosity statistics saved to scripts/output/region_porosity_stats.txt")

    create_porosity_profile(slice_porosities, slice_highlighted_voxels, stack.shape[0])

    return {
        "shape": stack.shape,
        "porosity": porosity,
        "min_porosity": min_porosity,
        "max_porosity": max_porosity,
        "mean_porosity": mean_porosity,
        "std_porosity": std_porosity,
        "highlighted_voxels": total_highlighted_voxels,
        "pore_voxels": pore_voxels,
    }


def create_porosity_profile(slice_porosities, slice_highlighted_voxels, total_slices):
    """
    Create a 2-panel plot showing:
      (1) porosity vs slice index
      (2) number of highlighted voxels per slice
    """
    valid_indices = [i for i, count in enumerate(slice_highlighted_voxels) if count > 0]

    if not valid_indices:
        print("Cannot create porosity profile: no slices with highlighted regions")
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Align porosity values to valid slice indices
    porosity_values = [slice_porosities[valid_indices.index(i)] for i in valid_indices]
    ax1.plot(valid_indices, porosity_values, "b-", marker="o", alpha=0.7)
    ax1.set_ylabel("Porosity (%)")
    ax1.set_title("Porosity Profile within Highlighted Regions")
    ax1.grid(True, alpha=0.3)

    # Plot size of highlighted regions
    nonzero_counts = [slice_highlighted_voxels[i] for i in valid_indices]
    ax2.bar(valid_indices, nonzero_counts, alpha=0.5, color="green")
    ax2.set_xlabel("Slice Index")
    ax2.set_ylabel("Highlighted Voxels")
    ax2.set_xlim(0, total_slices - 1)

    plt.tight_layout()
    plt.savefig("scripts/output/region_porosity_profile.png", dpi=300)
    print("Porosity profile saved to scripts/output/region_porosity_profile.png")
    plt.close()


def load_image_folder(folder_path: str) -> np.ndarray:
    """
    Load a folder of 2D image slices (PNG/TIFF/JPG) into a 3D stack [Z,H,W].
    Slices are sorted by filename.
    """
    folder = Path(folder_path)
    exts = {".ttif", ".tiff", ".png", ".jpg", ".jpeg"}

    files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    )
    if not files:
        raise FileNotFoundError(f"No image files found in folder: {folder_path}")

    slices = []
    for f in files:
        suffix = f.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            img = tifffile.imread(str(f))  # raw TIFF
        else:
            # PNG/JPG → read via PIL and convert to grayscale
            img = Image.open(f).convert("L")
            img = np.array(img)

        if img.ndim == 3:  # RGB fallback
            img = img[..., 0]

        slices.append(img)

    stack = np.stack(slices, axis=0)
    print(f"Loaded {len(files)} slices from folder {folder_path}, stack shape {stack.shape}")
    return stack


def load_stack(path):
    """
    Load an image stack with flexible path handling.

    Supports:
      - .npy volume
      - 3D .tif/.tiff stack
      - folder of 2D PNG/TIFF/JPG slices
    """
    # Try direct path first
    if os.path.exists(path):
        if os.path.isdir(path):
            # Folder of slices
            return load_image_folder(path)

        # Single file
        if path.endswith(".npy"):
            return np.load(path)
        else:
            return tifffile.imread(path)

    # Try common fallback directories
    filename = os.path.basename(path)
    common_locations = [
        os.path.join("scripts/output", filename),
        os.path.join("output", filename),
        os.path.join("data/pFIB", filename),
    ]

    for location in common_locations:
        if os.path.exists(location):
            if os.path.isdir(location):
                return load_image_folder(location)
            if location.endswith(".npy"):
                return np.load(location)
            else:
                return tifffile.imread(location)

    raise FileNotFoundError(f"Could not find {filename} in any common directory")


def main():
    parser = argparse.ArgumentParser(description="Calculate porosity of image stacks")
    parser.add_argument("stack_file", help="Path to image stack (TIFF/PNG folder, 3D TIFF, or NPY)")
    parser.add_argument(
        "highlight_file",
        nargs="?",
        default=None,
        help="Optional: highlighted region mask (same shape as stack)",
    )
    parser.add_argument(
        "--pores-are-black",
        "-b",
        action="store_true",
        help="Interpret black/dark areas as pores (default: white = pores)",
    )
    parser.add_argument(
        "--label",
        "-l",
        type=str,
        default=None,
        help="Custom label for the analysis",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Threshold factor for grayscale porosity (0–1, default 0.8)",
    )
    parser.add_argument(
        "--no-chdir",
        action="store_true",
        help="Do not change working directory to project root",
    )

    args = parser.parse_args()

    if not args.no_chdir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        os.chdir(project_root)
        print(f"Working directory: {os.getcwd()}")

    os.makedirs("scripts/output", exist_ok=True)

    # Default label from filename
    if args.label is None:
        args.label = os.path.basename(args.stack_file).split(".")[0].capitalize()

    try:
        print(f"Loading image stack from {args.stack_file}...")
        stack = load_stack(args.stack_file)
        print(f"Loaded stack with shape: {stack.shape}, dtype: {stack.dtype}")

        if args.highlight_file is None:
            print("No highlight file provided. Analyzing entire stack...")
            analyze_porosity(stack, label=args.label)
        else:
            print(f"Loading highlighted regions from {args.highlight_file}...")
            highlighted = load_stack(args.highlight_file)
            print(f"Loaded highlighted mask with shape: {highlighted.shape}, dtype: {highlighted.dtype}")

            # Crop if mismatched shapes
            if stack.shape != highlighted.shape:
                print("Warning: Stack and highlighted mask have different shapes")
                print(f"Stack: {stack.shape}, Highlighted: {highlighted.shape}")

                min_z = min(stack.shape[0], highlighted.shape[0])
                min_y = min(stack.shape[1], highlighted.shape[1])
                min_x = min(stack.shape[2], highlighted.shape[2])

                stack = stack[:min_z, :min_y, :min_x]
                highlighted = highlighted[:min_z, :min_y, :min_x]
                print(f"Cropped to common shape: {stack.shape}")

            pore_is_white = not args.pores_are_black
            print(f"Analyzing with {'black' if args.pores_are_black else 'white'} as pores...")

            analyze_region_porosity(
                stack,
                highlighted,
                label=args.label,
                pore_is_white=pore_is_white,
                threshold_factor=args.threshold,
            )

    except Exception as e:
        print(f"Error analyzing stack: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
