import numpy as np
from src import heatmap as hm
from src import kymograph as kg
from src import tiffclass as tiff
import optical_flow_cli as ofc

def show_heatmaps(flow, normalize=True):
    """
    Computes magnitude heatmaps from a flow array of shape (frames, height, width, 2).

    Args:
        flow (np.ndarray): Array of shape (frames, height, width, 2) with (dx, dy) vectors.
        apply_colormap (bool): If True, returns heatmaps with color (BGR, 3-channel).
        normalize (bool): If True, normalizes magnitudes to 0–255 range for visualization.

    Returns:
        None, just visualizes the heatmap
    """
    my_heatmap = hm.generate_heatmaps(flow)


def show_kymograph(line, ax=None, figsize=(10, 6), aspect='auto', cmap='PRGn', origin='upper', label='Kymograph', 
                   xlabel='Position along line', ylabel='Time (frame index)', title='Kymograph', show=True, save_path=None):
     """
    Plots a kymograph from a 2D array.

    Args:
        line (np.ndarray): 2D array representing the kymograph data.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates a new figure.
        figsize (tuple): Size of the figure in inches (width, height).
        aspect (str): Aspect ratio of the plot. Default is 'auto'.
        cmap (str): Colormap to use for the kymograph. Default is 'PRGn'.
        origin (str): Origin of the plot. Default is 'upper'.
        label (str): Label for the colorbar.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        show (bool): Whether to display the plot immediately.
    
    Returns:
        None: Just displays the kymograph.
    """
     my_kymograph = kg.plot_kymograph(line)

def show_tiff_image(Tiff: tiff.Tiff, image: np.ndarray, title='Image', figsize=(12, 8), save_path=None) -> None:
    """
    Displays or saves an image using matplotlib.

    Args:
        image (np.ndarray): Image to display.
        title (str): Title of the window.
        figsize (tuple): Figure size in inches (width, height).
        save_path (str, optional): If provided, saves the image to this path.

    Assumptions:
        The integers in the 'figsize' tuple are greater than 0.

    Returns:
        None
    """
    tiff.Tiff.show_image(image)

def show_nuclei_flow(arr: np.ndarray, channel: int) -> np.ndarray:
    """
    This function preprocess the tiff stack with the parameters.

    Args:
      arr: The stack from the initialized tiff class
    
    Returns:
      The optical flow array. 
      """
    raise NotImplementedError

def show_combined_flow(arr: np.ndarray) -> np.ndarray:
    """
    This function preprocess the tiff stack with the parameters.

    Args:
      arr: The stack from the initialized tiff class
    
    Returns:
      The optical flow array from combining the channels. 
      """
    return NotImplementedError

def show_raft_optical_flow(arr: np.ndarray) -> np.ndarray:
    """
    This function preprocess the tiff stack with the parameters.

    Args:
      arr: The stack from the initialized tiff class.
    
    Returns:
      The raft flow for the third channel
      """
    return NotImplementedError

     