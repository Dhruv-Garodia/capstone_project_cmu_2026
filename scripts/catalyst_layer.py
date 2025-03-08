import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from scipy.stats import lognorm
from scipy.optimize import curve_fit
from skimage import filters, morphology, feature
import porespy as ps
import argparse
import os

# 1. First, let's create synthetic 3D data to represent a small volume of catalyst layer
# (Normally, you would load actual pFIB-SEM data)

def generate_synthetic_catalyst_layer(size=100, porosity=0.5, noise_level=0.2):
    """Generate synthetic 3D data to represent a catalyst layer."""
    # Create a random 3D volume
    raw_volume = np.random.rand(size, size, size//2)  # Thinner in z-direction like a catalyst layer
    
    # Add some structure by applying Gaussian filter
    raw_volume = ndimage.gaussian_filter(raw_volume, sigma=1.0)
    
    # Threshold to get initial binary structure
    binary_volume = raw_volume < porosity
    
    # Add noise
    noise = np.random.rand(*binary_volume.shape) < noise_level
    binary_volume = np.logical_xor(binary_volume, noise)
    
    # Simulate some larger pore structures
    for _ in range(20):
        x = np.random.randint(0, binary_volume.shape[0]-10)
        y = np.random.randint(0, binary_volume.shape[1]-10)
        z = np.random.randint(0, binary_volume.shape[2]-10)
        radius = np.random.randint(3, 8)
        rr, cc, zz = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
        sphere = rr**2 + cc**2 + zz**2 <= radius**2
        try:
            binary_volume[x:x+2*radius+1, y:y+2*radius+1, z:z+2*radius+1][sphere] = True
        except:
            pass  # Handle edge cases
    
    return binary_volume

# 2. Implementation of the 6-step segmentation method

def segment_volume(volume):
    """Apply the 6-step segmentation process described in the paper."""
    # Step 1: Gaussian smoothing filter
    smoothed = ndimage.gaussian_filter(volume.astype(float), sigma=1.0)
    
    # Step 2: Forward gradient threshold (simplified version)
    gradient_z = ndimage.sobel(smoothed, axis=2)  # Assuming z-axis (axis=2) is direction of streak artifacts
    gradient_magnitude = np.abs(gradient_z)
    gradient_threshold = filters.threshold_otsu(gradient_magnitude)
    gradient_mask = gradient_magnitude > gradient_threshold
    
    # Step 3: Low-grayscale-value-removal
    low_threshold = 0.2 * np.max(smoothed)
    low_mask = smoothed < low_threshold
    
    # Step 4: High-grayscale-value-addition
    high_threshold = 0.8 * np.max(smoothed)
    high_mask = smoothed > high_threshold
    
    # Combine masks from steps 2-4
    binary = np.logical_or(gradient_mask, high_mask)
    binary = np.logical_and(binary, ~low_mask)
    
    # Step 5: 3D spherical element for opening and closing operations
    selem = morphology.ball(2)
    binary = morphology.opening(binary, selem)
    binary = morphology.closing(binary, selem)
    
    # Step 6: Final 3D Gaussian smoothing and re-thresholding
    final_smooth = ndimage.gaussian_filter(binary.astype(float), sigma=0.5)
    final_binary = final_smooth > 0.5
    
    return final_binary

# 3. Analysis functions

def calculate_porosity(binary_volume):
    """Calculate porosity as the fraction of pore voxels."""
    pore_voxels = np.sum(binary_volume)
    total_voxels = binary_volume.size
    return pore_voxels / total_voxels

def pore_size_distribution(binary_volume):
    """Calculate pore size distribution using PoreSpy local thickness algorithm."""
    # Use PoreSpy's porosimetry function to calculate pore sizes
    print("Calculating local thickness (this may take a moment)...")
    psd = ps.filters.local_thickness(binary_volume)
    
    # Get distribution of pore sizes (in voxels)
    unique_sizes = np.unique(psd[psd > 0])
    # Filter out extremely small sizes that might cause fitting issues
    unique_sizes = unique_sizes[unique_sizes >= 1.0]
    
    if len(unique_sizes) == 0:
        print("Warning: No valid pore sizes found.")
        return np.array([0]), np.array([0])
    
    counts = np.array([np.sum(psd == size) for size in unique_sizes])
    
    # Normalize the distribution
    if np.sum(counts) > 0:
        normalized_counts = counts / np.sum(counts)
    else:
        normalized_counts = counts
    
    return unique_sizes, normalized_counts

def fit_lognormal(sizes, distribution):
    """Fit log-normal distribution to the PSD data."""
    if len(sizes) < 3 or np.all(distribution == 0):
        print("Warning: Not enough data points for fitting.")
        return None, None, 0
    
    def lognorm_pdf(x, mu, sigma):
        """Log-normal probability density function."""
        return (1/(x*sigma*np.sqrt(2*np.pi))) * np.exp(-(np.log(x) - mu)**2 / (2*sigma**2))
    
    # Initial guess for parameters
    try:
        mean_size = np.mean(sizes)
        if mean_size <= 0:
            mean_size = 1.0
        p0 = [np.log(mean_size), 0.5]
        
        # Curve fitting with bounds to ensure positive values
        params, _ = curve_fit(
            lognorm_pdf, 
            sizes, 
            distribution, 
            p0=p0,
            bounds=([0, 0.01], [10, 5]), 
            maxfev=10000
        )
        
        mu, sigma = params
        y_fit = lognorm_pdf(sizes, mu, sigma)
        
        # Calculate R-squared
        ss_res = np.sum((distribution - y_fit)**2)
        ss_tot = np.sum((distribution - np.mean(distribution))**2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return mu, sigma, r_squared
    
    except Exception as e:
        print(f"Fitting error: {e}")
        return None, None, 0

# 4. Visualization functions

def visualize_3d_structure(binary_volume, title="3D Catalyst Layer Structure"):
    """Create a 3D visualization of the binary volume."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get coordinates of solid voxels
    z, y, x = np.where(binary_volume)
    
    # Subsample points if there are too many (for performance)
    max_points = 10000
    if len(x) > max_points:
        idx = np.random.choice(len(x), max_points, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
    
    # Plot points with alpha to see structure
    ax.scatter(x, y, z, c='darkblue', alpha=0.05, marker='.')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig

def plot_psd_with_fit(sizes, distribution, mu, sigma):
    """Plot PSD data with the log-normal fit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot original data
    ax.scatter(sizes, distribution, c='blue', label='PSD Data')
    
    # Plot fitted curve if parameters are available
    if mu is not None and sigma is not None:
        x_fit = np.linspace(max(1, min(sizes)), max(sizes), 100)
        y_fit = (1/(x_fit*sigma*np.sqrt(2*np.pi))) * np.exp(-(np.log(x_fit) - mu)**2 / (2*sigma**2))
        ax.plot(x_fit, y_fit, 'r-', label=f'Log-Normal Fit (μ={mu:.3f}, σ={sigma:.3f})')
    else:
        ax.text(0.5, 0.5, "Log-normal fitting failed", transform=ax.transAxes, 
                ha='center', fontsize=12, color='red')
    
    ax.set_xlabel('Pore Diameter [voxels]')
    ax.set_ylabel('Normalized Volume Fraction [-]')
    ax.set_title('Pore Size Distribution with Log-Normal Fit')
    ax.legend()
    
    plt.tight_layout()
    return fig

# 5. Generate 3D volume from known parameters

def generate_3d_from_parameters(size=(100, 100, 50), porosity=0.5, mu=4.6, sigma=0.6):
    """Generate a 3D structure from porosity and log-normal parameters."""
    # Create empty volume
    volume = np.zeros(size, dtype=bool)
    
    # Determine number of pores needed to achieve target porosity
    total_voxels = np.prod(size)
    target_pore_voxels = int(porosity * total_voxels)
    
    # Generate pore sizes from log-normal distribution
    num_pores = total_voxels // 100  # Arbitrary number of pores to distribute
    pore_sizes = np.random.lognormal(mean=mu, sigma=sigma, size=num_pores)
    pore_sizes = np.clip(pore_sizes, 1, min(size) // 3).astype(int)
    
    # Place pores randomly until target porosity is reached
    current_pore_voxels = 0
    pore_index = 0
    max_iterations = min(len(pore_sizes), 5000)  # Limit iterations for performance
    
    print(f"Placing pores to achieve {porosity:.3f} porosity...")
    
    for _ in range(max_iterations):
        if current_pore_voxels >= target_pore_voxels:
            break
            
        # Get next pore size
        if pore_index >= len(pore_sizes):
            pore_index = 0
            np.random.shuffle(pore_sizes)
            
        radius = max(1, pore_sizes[pore_index] // 2)
        pore_index += 1
            
        # Random position for pore center
        x = np.random.randint(radius, size[0] - radius)
        y = np.random.randint(radius, size[1] - radius)
        z = np.random.randint(radius, size[2] - radius)
        
        # Create sphere
        rr, cc, zz = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
        sphere = rr**2 + cc**2 + zz**2 <= radius**2
        
        # Add pore
        try:
            x_range = slice(max(0, x-radius), min(size[0], x+radius+1))
            y_range = slice(max(0, y-radius), min(size[1], y+radius+1))
            z_range = slice(max(0, z-radius), min(size[2], z+radius+1))
            
            # Get the correct part of the sphere that fits in the volume
            sphere_x = slice(max(0, radius-x), min(2*radius+1, size[0]-(x-radius)))
            sphere_y = slice(max(0, radius-y), min(2*radius+1, size[1]-(y-radius)))
            sphere_z = slice(max(0, radius-z), min(2*radius+1, size[2]-(z-radius)))
            
            # Update the volume
            volume[x_range, y_range, z_range] = np.logical_or(
                volume[x_range, y_range, z_range],
                sphere[sphere_x, sphere_y, sphere_z]
            )
            
            current_pore_voxels = np.sum(volume)
            
        except Exception as e:
            # Just continue with next iteration if there's any issue
            pass
    
    # Calculate actual achieved porosity
    actual_porosity = np.sum(volume) / total_voxels
    print(f"Target porosity: {porosity:.3f}, Achieved porosity: {actual_porosity:.3f}")
    
    return volume

# 6. Main execution - demonstrating the full workflow

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Generate and analyze 3D catalyst layer structure')
    
    # Required argument for porosity
    parser.add_argument('--porosity', type=float, required=True, 
                        help='Target porosity value (0-1)')
    
    # Optional arguments with defaults from the paper
    parser.add_argument('--region', type=str, default='custom',
                        choices=['Pristine', 'PTL-pore-area', 'PTL-fiber-compressed', 'custom'],
                        help='Region type from the paper or custom')
    
    parser.add_argument('--mu', type=float, default=None,
                        help='Log-normal mean parameter (default: based on region)')
    
    parser.add_argument('--sigma', type=float, default=None,
                        help='Log-normal standard deviation parameter (default: based on region)')
    
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Reference parameters from the paper
    log_normal_params = {
        "Pristine": (4.657, 0.6352),
        "PTL-pore-area": (4.926, 0.7396),
        "PTL-fiber-compressed": (4.572, 0.6127)
    }
    
    # Set mu and sigma based on region or user input
    if args.region != 'custom' and args.mu is None and args.sigma is None:
        mu, sigma = log_normal_params[args.region]
        print(f"Using parameters for {args.region} region:")
    else:
        # For custom, use provided values or defaults
        mu = args.mu if args.mu is not None else 4.6
        sigma = args.sigma if args.sigma is not None else 0.6
        print(f"Using custom parameters:")
    
    print(f"  - Porosity: {args.porosity:.3f}")
    print(f"  - Log-normal μ: {mu:.3f}")
    print(f"  - Log-normal σ: {sigma:.3f}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Generate 3D structure from parameters
    print("\nGenerating 3D structure...")
    volume = generate_3d_from_parameters(size=(100, 100, 30), 
                                         porosity=args.porosity, 
                                         mu=mu, 
                                         sigma=sigma)
    
    # Analyze the generated structure
    print("\nAnalyzing generated structure...")
    measured_porosity = calculate_porosity(volume)
    print(f"Measured porosity: {measured_porosity:.3f}")
    
    sizes, dist = pore_size_distribution(volume)
    measured_mu, measured_sigma, r2 = fit_lognormal(sizes, dist)
    
    print(f"Fitted log-normal parameters:")
    if measured_mu is not None and measured_sigma is not None:
        print(f"  - μ: {measured_mu:.3f} (target: {mu:.3f})")
        print(f"  - σ: {measured_sigma:.3f} (target: {sigma:.3f})")
        print(f"  - R²: {r2:.3f}")
    else:
        print("  - Fitting failed. Using target parameters for visualization.")
        measured_mu, measured_sigma = mu, sigma
    
    # Visualize results
    print("\nVisualizing results...")
    
    # 3D structure visualization
    region_name = args.region if args.region != 'custom' else f"Custom_P{args.porosity:.2f}"
    fig1 = visualize_3d_structure(volume, title=f"3D {region_name} Catalyst Layer Structure")
    fig1.savefig(os.path.join(args.output, f"{region_name}_3d_structure.png"), dpi=300)
    
    # PSD with log-normal fit
    fig2 = plot_psd_with_fit(sizes, dist, measured_mu, measured_sigma)
    fig2.savefig(os.path.join(args.output, f"{region_name}_psd_with_fit.png"), dpi=300)
    
    print(f"\nCompleted! Output images saved to {args.output}/")

if __name__ == "__main__":
    main()