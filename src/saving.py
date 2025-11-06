import os
import json
import numpy as np
from pathlib import Path
import matplotlib.animation as animation
from src.defaults import default_process, default_flow, default_trajectory

def save_arr(name: str, arr: np.ndarray) -> None:
    """
    Saves a numpy array to a file.
    
    Args:
        arr (np.ndarray): The numpy array to save.
    
    Returns:
        None: Just saves the array to a file.
    """
    return NotImplementedError

def save_flow(name: str, arr: np.ndarray):
    """
    Saves the optical flow array.
    
    Args:
        name (str): The name of the file.
        arr (np.ndarray): The optical flow or trajectory array to save, expected to be of shape (T, H, W, 2)
            where T is the number of frames, H is height, W is width, and the last dimension contains
            the flow vectors (dx, dy) or trajectory vectors.
    
    Returns:
        None: Just saves the array to a file.
    """
    return NotImplementedError

def save_trajectory(name: str, ftag: str, arr: np.ndarray) -> None:
    """
    Saves the trajectory flow array.

    Args:
        name (str): The name of the file.
        arr (np.ndarray): The optical flow or trajectory array to save, expected to be of shape (T, H, W, 2)
        ftag (str): The tag associated with the optical flow file the trajectory was derived from.
            - T is the number of frames
            - H is height
            - W is width
            - The last dimension contains the flow vectors (dx, dy) or trajectory vectors.
    
    Returns:
        None: Just saves the array to a file.
    """
    return NotImplementedError

def save_original_video(self, name: str, file_path: str, im: animation.Figure, image_stack, fig, ax, **kwargs) -> None:
    """
    Saves a video of image frames using matplotlib.

    Args:
        name (str): Name of the video file to save.
        file_path (str): The path to save the video file to.
        im (animation.Figure): Matplotlib image display object for the original frames.
        image_stack: Image stack of shape (T, H, W).
        fig: Matplotlib figure object for the plot.
        ax: Matplotlib axes object for the plot.
        **kwargs: Additional keyword arguments that include:
            - T (int): Total number of frames in the image stack.
            - fps (int): Frames per second for the video.

    Assumptions:
        'T' is greater than or equal to image_stack.shape[0].

    Returns:
        None: Just saves the video to the specified path.
    """
    T = kwargs.get('T', image_stack.shape[0])
    fps = kwargs.get('fps', 10)

    def update(frame):
        im.set_data(image_stack[frame])
        ax.set_title(f"Frame {frame}")

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(file_path, writer=writer)

def save_vector_video(name: str, flag: str, **kwargs) -> None:
    """
    Creates a video of optical flow vectors overlaid on the original image frames.

    Args:
        name (str): Name of the video file to save.
        flag (str): Flag to determine the type of video being saved.
        **kwargs: Additional keyword arguments that include:
            - img_disp: Matplotlib image display object for the original frames.
            - arr: Optical flow array of shape (T, H, W, 2) where T is the number of frames,
                   H is height, W is width, and the last dimension contains the flow vectors (dx, dy).
            - og_arr: Original image frames array of shape (T, H, W, C). Default is None.
            - step: Step size for downsampling the flow vectors for visualization. Default is 20.
            - fps: Frames per second for the video. Default is 10.
            - quiver: Matplotlib quiver object for displaying flow vectors.
            - ax: Matplotlib axes object for the plot.
            - fig: Matplotlib figure object for the plot.
            - T_minus_1: Total number of frames minus one (T-1).
    Returns:
        None: Just saves the video to the specified path.

    Invariant:
        Assumes, that all values are present in kwargs and are of the correct type. The check occurs in the
        `create_optical_flow_video` function (in TiffVisualize.py) before this function is called.
    """
    return NotImplementedError

def load_params(stacktype: str) -> dict:
    """
    Loads parameters from types.json.

    Args:
        stacktype (str): The type of cell.
    
    Returns:
        params: Dictionary of parameters.
    """
    return NotImplementedError