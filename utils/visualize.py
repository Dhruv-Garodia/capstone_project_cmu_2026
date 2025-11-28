import os
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import imageio.v2 as imageio

def load_any(path):
    """Load npy, tiff, or PNG directory."""
    if os.path.isdir(path):
        print(f"Detected PNG folder: {path}")
        return load_png_folder(path)

    if path.endswith(".npy"):
        return np.load(path)

    # Otherwise assume TIFF-like
    return tifffile.imread(path)

def load_png_folder(path):
    """Load a folder of PNGs into a 3D stack [Z, H, W]."""
    files = sorted([
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(".png")
    ])
    if len(files) == 0:
        raise ValueError(f"No PNG files found in directory: {path}")

    slices = []
    for p in files:
        img = imageio.imread(p)
        if img.ndim == 3:      # RGB → grayscale
            img = img[..., 0]
        slices.append(img)

    return np.stack(slices, axis=0)

def visualize_stack(data, title="Slice", cmap='gray'):
    """
    Visualizes a 3D image stack with a slider to navigate slices.
    
    Parameters:
        data (ndarray): 3D numpy array of the image stack
        title (str): Title prefix for the displayed slice
        cmap (str): Colormap to use for visualization
    """
    # Create a figure and axis for the image display
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)
    
    # Set initial slice index (middle of the volume)
    slice_index = data.shape[0] // 2
    im = ax.imshow(data[slice_index], cmap=cmap)
    ax.set_title(f"{title} {slice_index}")
    
    # Add information about the dataf
    info_text = f"Shape: {data.shape}\n"
    if data.dtype == bool:
        info_text += f"Type: boolean\n"
        info_text += f"Porosity (slice): {np.mean(data[slice_index]):.4f}\n"
        info_text += f"Porosity (volume): {np.mean(data):.4f}"
    else:
        info_text += f"Type: {data.dtype}\n"
        info_text += f"Min: {np.min(data[slice_index])}, Max: {np.max(data[slice_index])}\n"
        info_text += f"Mean: {np.mean(data[slice_index]):.2f}, Std: {np.std(data[slice_index]):.2f}"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Create slider axis and slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, data.shape[0]-1, valinit=slice_index, valstep=1)
    
    # Update function to change displayed slice based on slider value
    def update(val):
        idx = int(slider.val)
        im.set_data(data[idx])
        ax.set_title(f"{title} {idx}")
        
        # Update info text for current slice
        if data.dtype == bool:
            slice_porosity = np.mean(data[idx])
            plt.figtext(0.02, 0.02, 
                        f"Shape: {data.shape}\nType: boolean\n"
                        f"Porosity (slice): {slice_porosity:.4f}\n"
                        f"Porosity (volume): {np.mean(data):.4f}", 
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        else:
            plt.figtext(0.02, 0.02, 
                        f"Shape: {data.shape}\nType: {data.dtype}\n"
                        f"Min: {np.min(data[idx])}, Max: {np.max(data[idx])}\n"
                        f"Mean: {np.mean(data[idx]):.2f}, Std: {np.std(data[idx]):.2f}", 
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()

def visualize_side_by_side(data1, data2, title1="Original", title2="Processed"):
    """
    Visualize two stacks side by side with a shared slice slider.
    
    Parameters:
        data1, data2 (ndarray): 3D numpy arrays to compare
        title1, title2 (str): Titles for the two displays
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    plt.subplots_adjust(bottom=0.25)
    
    # Calculate initial slice index (middle of the volumes)
    slice_index = min(data1.shape[0], data2.shape[0]) // 2
    
    # Display initial slices
    im1 = ax1.imshow(data1[slice_index], cmap='gray')
    ax1.set_title(f"{title1} - Slice {slice_index}")
    
    im2 = ax2.imshow(data2[slice_index], cmap='gray')
    ax2.set_title(f"{title2} - Slice {slice_index}")
    
    # Add information about data1
    info_text1 = f"Shape: {data1.shape}\n"
    if data1.dtype == bool:
        info_text1 += f"Type: boolean\n"
        info_text1 += f"Porosity (slice): {np.mean(data1[slice_index]):.4f}\n"
        info_text1 += f"Porosity (volume): {np.mean(data1):.4f}"
    else:
        info_text1 += f"Type: {data1.dtype}\n"
        info_text1 += f"Min: {np.min(data1[slice_index])}, Max: {np.max(data1[slice_index])}\n"
        info_text1 += f"Mean: {np.mean(data1[slice_index]):.2f}, Std: {np.std(data1[slice_index]):.2f}"
    
    # Add information about data2
    info_text2 = f"Shape: {data2.shape}\n"
    if data2.dtype == bool:
        info_text2 += f"Type: boolean\n"
        info_text2 += f"Porosity (slice): {np.mean(data2[slice_index]):.4f}\n"
        info_text2 += f"Porosity (volume): {np.mean(data2):.4f}"
    else:
        info_text2 += f"Type: {data2.dtype}\n"
        info_text2 += f"Min: {np.min(data2[slice_index])}, Max: {np.max(data2[slice_index])}\n"
        info_text2 += f"Mean: {np.mean(data2[slice_index]):.2f}, Std: {np.std(data2[slice_index]):.2f}"
    
    text1 = plt.figtext(0.02, 0.02, info_text1, fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8))
    text2 = plt.figtext(0.52, 0.02, info_text2, fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8))
    
    # Create slice slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    max_slice = min(data1.shape[0], data2.shape[0]) - 1
    slider = Slider(ax_slider, 'Slice', 0, max_slice, valinit=slice_index, valstep=1)
    
    # Update function for slider
    def update(val):
        idx = int(slider.val)
        
        # Update images
        if idx < data1.shape[0]:
            im1.set_data(data1[idx])
            ax1.set_title(f"{title1} - Slice {idx}")
            
            # Update data1 info
            if data1.dtype == bool:
                slice_porosity1 = np.mean(data1[idx])
                new_text1 = f"Shape: {data1.shape}\nType: boolean\n" \
                           f"Porosity (slice): {slice_porosity1:.4f}\n" \
                           f"Porosity (volume): {np.mean(data1):.4f}"
            else:
                new_text1 = f"Shape: {data1.shape}\nType: {data1.dtype}\n" \
                           f"Min: {np.min(data1[idx])}, Max: {np.max(data1[idx])}\n" \
                           f"Mean: {np.mean(data1[idx]):.2f}, Std: {np.std(data1[idx]):.2f}"
            text1.set_text(new_text1)
            
        if idx < data2.shape[0]:
            im2.set_data(data2[idx])
            ax2.set_title(f"{title2} - Slice {idx}")
            
            # Update data2 info
            if data2.dtype == bool:
                slice_porosity2 = np.mean(data2[idx])
                new_text2 = f"Shape: {data2.shape}\nType: boolean\n" \
                           f"Porosity (slice): {slice_porosity2:.4f}\n" \
                           f"Porosity (volume): {np.mean(data2):.4f}"
            else:
                new_text2 = f"Shape: {data2.shape}\nType: {data2.dtype}\n" \
                           f"Min: {np.min(data2[idx])}, Max: {np.max(data2[idx])}\n" \
                           f"Mean: {np.mean(data2[idx]):.2f}, Std: {np.std(data2[idx]):.2f}"
            text2.set_text(new_text2)
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Visualize TIFF or NPY image stacks')
    parser.add_argument('file_paths', nargs='*', help='Paths to image stack files (one or two)')
    parser.add_argument('--title1', '-t1', type=str, help='Title for first image')
    parser.add_argument('--title2', '-t2', type=str, help='Title for second image')
    parser.add_argument('--no-chdir', action='store_true', help='Do not change to project root directory')
    
    args = parser.parse_args()
    
    # Change to project root directory if not disabled
    if not args.no_chdir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        os.chdir(project_root)
        print(f"Changed working directory to: {os.getcwd()}")
    
    # Get file paths from arguments
    if not args.file_paths:
        print("Error: No file paths provided.")
        parser.print_help()
        return
    
    # Process file paths
    file_paths = args.file_paths
    if len(file_paths) > 2:
        print("Warning: More than two files provided. Only the first two will be used.")
        file_paths = file_paths[:2]
    
    # Determine titles
    titles = [os.path.basename(path) for path in file_paths]
    if args.title1 and len(file_paths) >= 1:
        titles[0] = args.title1
    if args.title2 and len(file_paths) >= 2:
        titles[1] = args.title2
    
    # Load files
    try:
        # Try to load first file
        print(f"Loading {file_paths[0]}...")
        data1 = load_any(file_paths[0])
        print(f"Loaded {file_paths[0]} with shape: {data1.shape}, dtype: {data1.dtype}")
        
        # If two files are provided, load the second file and display side by side
        if len(file_paths) == 2:
            print(f"Loading {file_paths[1]}...")
            data2 = load_any(file_paths[1])
            print(f"Loaded {file_paths[1]} with shape: {data2.shape}, dtype: {data2.dtype}")
            visualize_side_by_side(data1, data2, title1=titles[0], title2=titles[1])
        else:
            visualize_stack(data1, title=titles[0])
            
    except Exception as e:
        print(f"Error loading or visualizing file(s): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()