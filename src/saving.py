import os
import json
import numpy as np
import pathlib
from pathlib import Path
import matplotlib.animation as animation
from src.defaults import default_process, default_flow, default_trajectory
import matplotlib.image, matplotlib.figure, matplotlib.axes

#UPDATE THESE
main_path = Path.cwd() / "CellFlow" # update this to make it desktop
inbox_path = main_path / "inbox"
types_path = main_path / "types.json"

def init_memory() -> None:
    """
    Initializes memory for the application.

    This creates a main folder 'OpticalFlow' on the user's Desktop, along with a 'types.json'
    file if it doesn't already exist.

    You can find a detailed description of this directory's structure under the README on Github.

    Args:
        None

    Returns:
        None
    """
    #CHECK IF ZIYAD HAS ALREADY WRITTEN THIS FUNCTION IN HIS OTHER REPO HE DID OVER THE SUMMER! IT MIGHT BE IN CELL_FLOW_TRACKING'S MEMORY.PY. BUT WE MIGHT HAVE TO EDIT IT
    return NotImplementedError

def get_unique_path(name, file_type, pattern_fn) -> Path:
    """
    Generates a unique file path in the given directory based on a naming pattern.

    Args:
        name (str): Main identifier (e.g., protein name).
        file_type (str): Subdirectory (e.g., 'flow', 'trajectory').
        pattern_fn (callable): Function that takes an integer and returns a file name.

    Returns:
        Path: Unique file path that does not yet exist.
    """
    #CHECK IF ZIYAD HAS ALREADY WRITTEN THIS FUNCTION IN HIS OTHER REPO HE DID OVER THE SUMMER! IT MIGHT BE IN CELL_FLOW_TRACKING'S MEMORY.PY. BUT WE MIGHT HAVE TO EDIT IT
    return NotImplementedError

def save_arr(name: str, arr: np.ndarray) -> None:
    """
    Saves a numpy array to a file.
    
    Args:
        arr (np.ndarray): The numpy array to save.
    
    Returns:
        None: Just saves the array to a file.
    """
    #Caroline
    #CHECK IF ZIYAD HAS ALREADY WRITTEN THIS FUNCTION IN HIS OTHER REPO HE DID OVER THE SUMMER! IT MIGHT BE IN CELL_FLOW_TRACKING'S MEMORY.PY. BUT WE MIGHT HAVE TO EDIT IT
    return NotImplementedError

def save_optical_flow_as_xyz() -> None:
    """
    Saves the optical flow array as an XYZ array.

    Args:
        ADD
    
    Returns:
        ADD
    """
    #Caroline
    return NotImplementedError

def save_optical_flow_as_matlab() -> None:
    """
    Saves the optical flow array as a MATLAB array.

    Args:
        ADD
    
    Returns:
        ADD
    """
    #Caroline
    return NotImplementedError

def save_optical_flow_as_numpy() -> None:
    """
    Saves the optical flow array as a numpy array.

    Args:
        ADD
    
    Returns:
        ADD
    """
    #Caroline
    return NotImplementedError

def save_flow(name: str, arr: np.ndarray):
    """
    Saves the optical flow array as an XYZ array, as a MATLAB array, and as a numpy array.
    
    Args:
        name (str): The name of the file.
        arr (np.ndarray): The optical flow or trajectory array to save, expected to be of shape (T, H, W, 2)
            where T is the number of frames, H is height, W is width, and the last dimension contains
            the flow vectors (dx, dy) or trajectory vectors.
    
    Returns:
        None: Just saves the array as an XYZ array, as a MATLAB array, and as a numpy array..
    """
    #Caroline
    #Call XYZ function, MATLAB function, and numpy function
    #Maybe not: CHECK IF ZIYAD HAS ALREADY WRITTEN THIS FUNCTION IN HIS OTHER REPO HE DID OVER THE SUMMER! IT MIGHT BE IN CELL_FLOW_TRACKING'S MEMORY.PY. BUT WE MIGHT HAVE TO EDIT IT
    return NotImplementedError

def save_original_video(name: str, file_path: pathlib.PosixPath, im: matplotlib.image.AxesImage, image_stack: np.ndarray, fig: matplotlib.figure.Figure, ax: matplotlib.axes._axes.Axes, **kwargs) -> None:
    """
    Saves a video of image frames using matplotlib.

    Args:
        name (str): Name of the video file to save.
        file_path (pathlib.PosixPath): The path to save the video file to.
        im (matplotlib.image.AxesImage): Matplotlib image display object for the original frames.
        image_stack (np.ndarray): Image stack of shape (T, H, W).
        fig (matplotlib.figure.Figure): Matplotlib figure object for the plot.
        ax (matplotlib.axes._axes.Axes): Matplotlib axes object for the plot.
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