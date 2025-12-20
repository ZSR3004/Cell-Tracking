import numpy as np
import os
from src.cell_tracking import heatmap as hm
from src.cell_tracking import kymograph as kg
from src.cell_tracking import vector_magnitude_map as vm
from . import optical_flow_cli as ofc


def plot_heatmap_cli(flow, title, output_path):
    """
    Computes magnitude heatmaps from a flow array of shape (frames, height, width, 2).

    Args:
        flow (np.ndarray): Array of shape (frames, height, width, 2) with (dx, dy) vectors.
        apply_colormap (bool): If True, returns heatmaps with color (BGR, 3-channel).
        normalize (bool): If True, normalizes magnitudes to 0–255 range for visualization.

    Returns:
        None, just visualizes the heatmap
    """
    hm.plot_heatmap(flow, title, output_path)


def plot_basic_kymo_cli(
    arr: np.ndarray,
    save_path: str,
    threshold=0.5,
):
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

def vector_video_cli(name, arr : np.ndarray, og_arr : np.ndarray=None, 
                    step : int = 20, scale : int = 500, color : str = 'blue', 
                    fps : int = 10, figsize : int | int = (12,8),
                    title : str = None, flag : str = None):
    """
    Saves a video of optical flow (quiver animation), optionally overlaid on image frames.

    Args:
        name (str): Name of the video file to save.
        arr (np.ndarray): Optical flow array of shape (T, H, W, 2) where T is the number of frames,
                          H is height, W is width, and the last dimension contains the flow vectors (dx, dy).
        og_arr (np.ndarray): Original image frames array of shape (T, H, W, C). Default is None.
        step (int): Step size for downsampling the flow vectors for visualization. Default is 20.
        scale (int): Scale factor for the quiver arrows. Default is 500.
        color (str): Color of the arrows. Default is 'blue'.
        fps (int): Frames per second for the video. Default is 10.
        figsize (tuple): Figure size in inches (width, height). Default is (12, 8).
        title (str): Title of the video. Default is None.
        flag (str): Flag to determine if the video should be saved ('f' for flow, 't' for trajectory). Default is None.

    Returns:
        None
    """
    vm.create_vector_field_video(name, arr, og_arr, step, scale, color, fps, figsize, title, flag)
     
