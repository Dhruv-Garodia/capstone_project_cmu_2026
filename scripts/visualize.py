import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_stack(data, title="Slice"):
    """
    Visualizes a 3D image stack with a slider to navigate slices.
    
    Parameters:
        data (ndarray): 3D numpy array of the segmented volume.
        title (str): Title prefix for the displayed slice.
    """
    # Create a figure and axis for the image display
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    
    # Set initial slice index (middle of the volume)
    slice_index = data.shape[0] // 2
    im = ax.imshow(data[slice_index], cmap='gray')
    ax.set_title(f"{title} {slice_index}")
    
    # Create slider axis and slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, data.shape[0]-1, valinit=slice_index, valstep=1)
    
    # Update function to change displayed slice based on slider value
    def update(val):
        idx = int(slider.val)
        im.set_data(data[idx])
        ax.set_title(f"{title} {idx}")
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()

def main():
    # Load segmented TIFF stack
    os.chdir("..")
    try:
        tiff_stack = tifffile.imread("data/pFIB/pristine_full.tif")
        print("Loaded pristine_full.tif with shape:", tiff_stack.shape)
        visualize_stack(tiff_stack, title="TIFF Stack Slice")
    except Exception as e:
        print("Error loading pristine_full.tif:", e)
    
    try:
        tiff_stack = tifffile.imread("scripts/output/segmented_stack.tif")
        print("Loaded segmented_stack.tif with shape:", tiff_stack.shape)
        visualize_stack(tiff_stack, title="TIFF Stack Slice")
    except Exception as e:
        print("Error loading segmented_stack.tif:", e)
    
    # Load segmented NumPy volume
    try:
        npy_volume = np.load("scripts/output/segmented_volume.npy")
        print("Loaded segmented_volume.npy with shape:", npy_volume.shape)
        visualize_stack(npy_volume, title="Numpy Volume Slice")
    except Exception as e:
        print("Error loading segmented_volume.npy:", e)

if __name__ == "__main__":
    main()
