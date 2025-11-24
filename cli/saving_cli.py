import os
from pathlib import Path
import sys
import matplotlib
# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.cell_tracking import saving
import numpy as np

def save_flow_cli(name: str,
                  opt_flow: np.ndarray, main_path: str):
    """
    Saves the raw data to the specified foler (raw data folder)

    Args:
        xyz_name: name of the xyz array
        matlab_name: name of the matlab array
        numpy_name: name of the numpy arrayj
        opt_flow: the optical flow array to save
        main_path: the path to the file to save

    Returns:
        None: just saves the arrays to the specified directories
    """

    main_path = Path(main_path)
    
    saving.save_optical_flow_as_xyz(name, opt_flow, main_path)
    saving.save_optical_flow_as_matlab(name, opt_flow, main_path)
    saving.save_optical_flow_as_numpy(name, opt_flow, main_path)

def save_arr_cli(arr : np.array) -> None:
    """
    Saves a numpy array to a file.
    
    Args:
        arr (np.array): The numpy array to save.
    
    Returns:
        None: Just saves the array to a file.
    """
    saving.save_arr(arr)


def save_original_video_cli(name: str, file_path: str) -> None:
    """
    Saves a video of image frames using matplotlib.
    Args:
        name (str): Name of the video file to save.
        **kwargs: Additional keyword arguments that include:
            - im: Matplotlib image display object for the original frames.
            - image_stack: Image stack of shape (T, H, W) or (T, H, W, 3) for RGB.
            - ax: Matplotlib axes object for the plot.
            - fig: Matplotlib figure object for the plot.
            - T: Total number of frames in the image stack.
            - fps: Frames per second for the video.
        Returns:
            None: Just saves the video to the specified path.
    """
    return saving.save_original_video(name, file_path, )

def save_vector_video_cli(name : str, flag : str, **kwargs) -> None:
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
    return saving.save_vector_video(name, flag, kwargs)