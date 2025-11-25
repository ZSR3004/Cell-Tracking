import numpy as np
import os
from src.cell_tracking import heatmap as hm
from src.cell_tracking import kymograph as kg
from src.cell_tracking import tiffclass as tiff
import optical_flow_cli as ofc

def save_heatmap_video_cli(flow, output_path, normalize=True):
    """
    Computes magnitude heatmaps from a flow array of shape (frames, height, width, 2).

    Args:
        flow (np.ndarray): Array of shape (frames, height, width, 2) with (dx, dy) vectors.
        apply_colormap (bool): If True, returns heatmaps with color (BGR, 3-channel).
        normalize (bool): If True, normalizes magnitudes to 0–255 range for visualization.

    Returns:
        None, just visualizes the heatmap
    """
    hm.save_heatmap_video(flow, output_path)


def plot_basic_kymo_cli(arr:np.ndarray, threshold=0.5, save_path = os.getcwd()):
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
    kg.plot_basic_kymo(arr, save_path)

     