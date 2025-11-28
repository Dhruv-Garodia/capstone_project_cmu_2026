#!/usr/bin/env python3
"""
Mesh -> 2D slices (PNG) -> pore size distribution (porespy) pipeline.

Usage example:
  python mesh_pore_pipeline.py \
      --mesh pfib_mesh.stl \
      --axis z \
      --n_slices 30 \
      --slice_dir slices_out_new \
      --analysis_dir pore_analysis_results_new

Required packages:
  pip install trimesh matplotlib numpy shapely \
              scikit-image scipy porespy pandas tqdm
"""

import os
import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import trimesh

from skimage import io, filters, morphology
from scipy import ndimage
import porespy as ps
from tqdm import tqdm


# ----------------------------------------------------------------------
# Mesh slicing utilities
# ----------------------------------------------------------------------

def path2d_to_polylines(path2d, scale=1.0):
    """
    Convert a trimesh Path2D to a list of polylines (each polyline is (N, 2) array).

    This uses trimesh.path.traversal.discretize_path to sample each path.
    """
    from trimesh.path import traversal

    polylines = []
    # path2d.paths: list of entity index sequences that form closed/open curves
    for path in path2d.paths:
        pts = traversal.discretize_path(
            path2d.entities,
            path2d.vertices,
            path,
            scale=scale,
        )
        polylines.append(pts)
    return polylines


def slice_mesh_to_png(
    mesh_path: str,
    axis: str = "z",
    n_slices: int = 30,
    margin: float = 1e-6,
    show_3d: bool = False,
    save_svg: bool = False,
    save_dxf: bool = False,
    save_png: bool = True,
    out_dir: Path = Path("slices_out_new"),
):
    """
    Slice a mesh along a chosen axis into multiple planes and export 2D contours.

    Returns
    -------
    slices_info : list of dict
        Each dict has keys: 'pos', 'path3d', 'path2d'
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(mesh_path)
    if mesh.is_empty:
        raise ValueError(f"Failed to load mesh or mesh is empty: {mesh_path}")

    axis = axis.lower()
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    normal = np.eye(3)[axis_idx]
    amin, amax = mesh.bounds[:, axis_idx]

    # Slice positions
    positions = np.linspace(amin + margin, amax - margin, n_slices)

    slices = []   # list of dicts: {"pos": float, "path3d": Path3D, "path2d": Path2D}
    for pos in positions:
        origin = mesh.bounds[0].copy()
        origin[axis_idx] = pos

        path3d = mesh.section(plane_origin=origin, plane_normal=normal)
        if path3d is None:
            continue

        path2d, _T = path3d.to_planar()
        slices.append({"pos": pos, "path3d": path3d, "path2d": path2d})

    print(f"[SLICER] Made {len(slices)} slice(s) on axis {axis}.")

    # Optional 3D visualization (opens a viewer if backend is available)
    if show_3d and len(slices):
        scene = trimesh.Scene()
        mesh_vis = mesh.copy()
        mesh_vis.visual.face_colors = [200, 200, 200, 100]
        scene.add_geometry(mesh_vis)
        for s in slices:
            scene.add_geometry(s["path3d"])
        scene.show()

    # 2D grid overview plot
    if len(slices):
        cols = min(5, len(slices))
        rows = int(np.ceil(len(slices) / cols))
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(3.0 * cols, 3.0 * rows),
                                 squeeze=False)
        for ax, s in zip(axes.ravel(), slices):
            polylines = path2d_to_polylines(s["path2d"])
            for poly in polylines:
                if len(poly) >= 2:
                    ax.plot(poly[:, 0], poly[:, 1], lw=1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{axis} = {s['pos']:.3f}")
            ax.axis("off")
        # hide unused axes
        for ax in axes.ravel()[len(slices):]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    # Export each slice
    if (save_svg or save_dxf or save_png) and len(slices):
        for i, s in enumerate(slices):
            stem = f"slice_{axis}_{i:03d}"
            if save_svg:
                svg = s["path2d"].to_svg()
                (out_dir / f"{stem}.svg").write_text(svg)
            if save_dxf:
                s["path2d"].export(out_dir / f"{stem}.dxf")
            if save_png:
                fig, ax = plt.subplots(figsize=(4, 4))
                polylines = path2d_to_polylines(s["path2d"])
                for poly in polylines:
                    if len(poly) >= 2:
                        ax.plot(poly[:, 0], poly[:, 1], lw=1)
                ax.set_aspect("equal", adjustable="box")
                ax.axis("off")
                fig.tight_layout(pad=0)
                fig.savefig(out_dir / f"{stem}.png", dpi=300)
                plt.close(fig)

    # Optional per-slice area metric (requires shapely)
    try:
        import shapely.geometry as _sg  # noqa: F401
        areas = []
        for s in slices:
            polys = s["path2d"].polygons_full
            area_sum = sum(p.area for p in polys) if polys else 0.0
            areas.append((s["pos"], area_sum))
        if areas:
            print("[SLICER] Example: first 5 slice areas (in plane units^2):")
            print(areas[:5])
    except Exception:
        pass

    return slices, out_dir


# ----------------------------------------------------------------------
# Pore analysis utilities
# ----------------------------------------------------------------------

def preprocess_image(image):
    """
    Preprocess image for pore analysis:
    - Convert to grayscale if needed
    - Normalize
    - Threshold (pores assumed dark)
    - Remove small objects/holes
    """
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    image = image / np.max(image)

    # Otsu threshold; pores assumed darker
    thresh = filters.threshold_otsu(image)
    binary = image < thresh

    # Clean small noise
    binary = morphology.remove_small_objects(binary, min_size=10)
    binary = morphology.remove_small_holes(binary, area_threshold=10)

    return binary


def calculate_pore_size_distribution(binary_image, bins=8):
    """
    Calculate pore size distribution using porespy.
    Returns (bin_centers, probability, thickness_map).
    """
    try:
        # Local thickness (map of pore diameters in voxels)
        dt = ps.filters.local_thickness(binary_image)
        print("Max thickness (px):", dt.max())

        psd = ps.metrics.pore_size_distribution(
            dt,
            bins=bins,
            log=False,
            voxel_size=10.0,  # adjust if you know real voxel size
        )
        bin_centers = psd.bin_centers
        probability = psd.pdf

        return bin_centers, probability, dt

    except Exception as e:
        print(f"[PSD] Error in PSD calculation (falling back to EDT): {e}")
        dt = ndimage.distance_transform_edt(binary_image)
        dt = dt * 2  # radius -> diameter

        hist, bin_edges = np.histogram(dt[dt > 0], bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return bin_centers, hist, dt


def process_single_image(image_path, output_dir, bins=8):
    """
    Process a single image and return (bin_centers, probability, stats).
    Also saves a per-image figure to output_dir.
    """
    try:
        image = io.imread(image_path)
        binary = preprocess_image(image)
        bin_centers, probability, thickness_map = calculate_pore_size_distribution(
            binary, bins=bins
        )

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Pore Analysis: {os.path.basename(image_path)}', fontsize=14)

        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Binary mask
        axes[0, 1].imshow(binary, cmap='gray')
        axes[0, 1].set_title('Binary (Pore Space)')
        axes[0, 1].axis('off')

        # Local thickness map
        im = axes[1, 0].imshow(thickness_map, cmap='viridis')
        axes[1, 0].set_title('Local Thickness Map')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], shrink=0.8)

        # PSD curve
        axes[1, 1].plot(bin_centers, probability, linewidth=2)
        axes[1, 1].set_xlabel('Pore Size (pixels)')
        axes[1, 1].set_ylabel('Probability Density')
        axes[1, 1].set_title('Pore Size Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        plot_name = f"psd_{os.path.splitext(os.path.basename(image_path))[0]}.png"
        plt.savefig(os.path.join(output_dir, plot_name),
                    dpi=300, bbox_inches='tight')
        plt.close()

        porosity = np.sum(binary) / binary.size
        mean_pore_size = np.average(bin_centers, weights=probability)

        stats = {
            'filename': os.path.basename(image_path),
            'porosity': porosity,
            'mean_pore_size': mean_pore_size,
            'max_pore_size': bin_centers[np.argmax(probability)],
            'total_pores': int(np.sum(binary)),
        }

        return bin_centers, probability, stats

    except Exception as e:
        print(f"[PSD] Error processing {image_path}: {e}")
        return None, None, None


def run_pore_analysis(
    input_folder: str,
    output_folder: str,
    bins: int = 8,
):
    """
    Run pore size analysis on all images in input_folder, save results to output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))

    if not image_files:
        print(f"[PSD] No images found in {input_folder}")
        return

    print(f"[PSD] Found {len(image_files)} images to process")

    all_bin_centers = []
    all_probabilities = []
    all_stats = []

    print("\n[PSD] Processing individual images...")
    for img_path in tqdm(image_files, desc="Processing images"):
        bin_centers, probability, stats = process_single_image(
            img_path, output_folder, bins=bins
        )
        if bin_centers is not None:
            all_bin_centers.append(bin_centers)
            all_probabilities.append(probability)
            all_stats.append(stats)

    if not all_stats:
        print("[PSD] No images were successfully processed!")
        return

    print("\n[PSD] Creating aggregated analysis...")

    # Interpolate all PSDs to common bin centers
    min_size = min([bc.min() for bc in all_bin_centers])
    max_size = max([bc.max() for bc in all_bin_centers])
    common_bins = np.linspace(min_size, max_size, 100)

    interpolated_psds = []
    for bc, prob in zip(all_bin_centers, all_probabilities):
        interp_prob = np.interp(common_bins, bc, prob)
        interpolated_psds.append(interp_prob)

    mean_psd = np.mean(interpolated_psds, axis=0)
    std_psd = np.std(interpolated_psds, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Aggregated Pore Size Distribution Analysis', fontsize=16)

    # Individual PSDs
    for bc, prob in zip(all_bin_centers, all_probabilities):
        axes[0, 0].plot(bc, prob, alpha=0.3, linewidth=1)
    axes[0, 0].set_xlabel('Pore Size (pixels)')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].set_title('Individual PSDs')
    axes[0, 0].grid(True, alpha=0.3)

    # Mean PSD with std band
    axes[0, 1].plot(common_bins, mean_psd, linewidth=3, label='Mean')
    axes[0, 1].fill_between(common_bins, mean_psd - std_psd, mean_psd + std_psd,
                            alpha=0.3, label='±1 std')
    axes[0, 1].set_xlabel('Pore Size (pixels)')
    axes[0, 1].set_ylabel('Probability Density')
    axes[0, 1].set_title('Mean PSD with Standard Deviation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    stats_df = pd.DataFrame(all_stats)

    # Porosity boxplot
    axes[1, 0].boxplot([stats_df['porosity']], labels=['Porosity'])
    axes[1, 0].set_ylabel('Porosity')
    axes[1, 0].set_title('Porosity Distribution')
    axes[1, 0].grid(True, alpha=0.3)

    # Mean pore size boxplot
    axes[1, 1].boxplot([stats_df['mean_pore_size']], labels=['Mean Pore Size'])
    axes[1, 1].set_ylabel('Mean Pore Size (pixels)')
    axes[1, 1].set_title('Mean Pore Size Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'aggregated_psd_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # Save statistical summary
    stats_df.to_csv(os.path.join(output_folder, 'pore_statistics.csv'),
                    index=False)

    # Save aggregated PSD data
    psd_data = pd.DataFrame({
        'bin_centers': common_bins,
        'mean_probability': mean_psd,
        'std_probability': std_psd,
    })
    psd_data.to_csv(os.path.join(output_folder, 'aggregated_psd_data.csv'),
                    index=False)

    print("\n[PSD] Analysis Complete!")
    print(f"[PSD] Results saved in: {output_folder}")
    print(f"[PSD] Processed {len(all_stats)} images successfully")
    print("\n[PSD] Summary Statistics:")
    print(f"  Mean Porosity: {stats_df['porosity'].mean():.4f} "
          f"± {stats_df['porosity'].std():.4f}")
    print(f"  Mean Pore Size: {stats_df['mean_pore_size'].mean():.2f} "
          f"± {stats_df['mean_pore_size'].std():.2f} pixels")
    print(f"  Range of Pore Sizes: {min_size:.2f} - {max_size:.2f} pixels")


# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Slice a mesh and compute pore size distributions on slices."
    )
    parser.add_argument("--mesh", type=str, default="pfib_mesh.stl",
                        help="Path to input mesh (STL/OBJ/PLY).")
    parser.add_argument("--axis", type=str, default="z",
                        choices=["x", "y", "z"],
                        help="Axis along which to slice the mesh.")
    parser.add_argument("--n_slices", type=int, default=30,
                        help="Number of slicing planes.")
    parser.add_argument("--slice_dir", type=str, default="slices_out_new",
                        help="Directory to save 2D slices (PNG/SVG/DXF).")
    parser.add_argument("--analysis_dir", type=str,
                        default="pore_analysis_results_new",
                        help="Directory to save pore analysis results.")
    parser.add_argument("--no_show_3d", action="store_true",
                        help="Disable 3D visualization of mesh + slices.")
    parser.add_argument("--no_svg", action="store_true",
                        help="Do not save SVG slices.")
    parser.add_argument("--no_dxf", action="store_true",
                        help="Do not save DXF slices.")
    parser.add_argument("--no_png", action="store_true",
                        help="Do not save PNG slices.")
    parser.add_argument("--bins", type=int, default=8,
                        help="Number of bins for PSD.")
    return parser.parse_args()


def main():
    args = parse_args()

    # save_svg = not args.no_svg
    # save_dxf = not args.no_dxf
    # save_png = not args.no_png
    # show_3d = not args.no_show_3d

    # print("[MAIN] Slicing mesh...")
    # _, slices_dir = slice_mesh_to_png(
    #     mesh_path=args.mesh,
    #     axis=args.axis,
    #     n_slices=args.n_slices,
    #     margin=1e-6,
    #     show_3d=show_3d,
    #     save_svg=save_svg,
    #     save_dxf=save_dxf,
    #     save_png=save_png,
    #     out_dir=Path(args.slice_dir),
    # )
    
    slices_dir = "test_out_single_300_nopadding"
    print("\n[MAIN] Running pore analysis on generated slices...")
    run_pore_analysis(
        input_folder=str(slices_dir),
        output_folder=args.analysis_dir,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
