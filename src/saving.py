import os
import json
import numpy as np
import pathlib
from pathlib import Path
import matplotlib.animation as animation
import scipy.io 
from scipy.io import savemat
from src import tiffclass as tiff
from src.defaults import default_process, default_flow, default_trajectory
import matplotlib.image, matplotlib.figure, matplotlib.axes


def get_unique_path(name: str, pattern_fn, main_path: str) -> Path:
    """
    Generates a unique file path in the given directory based on a naming pattern.

    Args:
        name (str): Main identifier (e.g., protein name).
        pattern_fn (callable): Function that takes an integer and returns a file name.
        main_path (str): Main path to the directory.

    Returns:
        Path: Unique file path that does not yet exist.
    """
    save_dir = main_path / name
    save_dir.mkdir(parents=True, exist_ok=True)

    i = 1
    while True:
        file_name = pattern_fn(i)
        file_path = save_dir / file_name
        if not file_path.exists():
            return file_path
        i += 1

def save_arr(name: str, tiff_instance: tiff.Tiff, main_path: str) -> None:
    """
    Saves a numpy array (from the Tiff class) to a file.
    
    Args:
        name (str): Name of the numpy array.
        tiff_instance (Tiff.tiff): Instance of the Tiff class.
        main_path (str): Main path to the directory.
    
    Returns:
        None: Just saves the array to a file.
    """
    tiff_arr = tiff_instance.arr
    save_path = get_unique_path(name, lambda i: f"{name}_f{i}.npy", main_path)
    np.save(save_path, tiff_arr)

def save_optical_flow_as_xyz(name: str, opt_flow: np.ndarray, main_path: str) -> None:
    """
    Saves the optical flow array as an XYZ file.

    Args:
        name (str): Name of the numpy array.
        opt_flow (np.ndarray): The optical flow array.
        main_path (str): Main path to the directory.
    
    Returns:
        None: Just saves the optical flow array to a file.
    """
    #Caroline
    #turn optical flow array into xyz file and then save it
    #first figure out the shape of the optical flow array. Ziyad says (frames, channel, height, width, 2) 
    #the 2 is a (dx, dy) tuple. You use a library called atomic xyz pipeline (PROBABLY NOT)
    #you need to save it to the hard drive. Do it using np.save()
    #since (dx, dy) only concerns x and y, we need to deal with z. Here's how you do it: the XYZ array is (dx, dy, 0)
    #probably use/import the library xyz-py. Save using this: xyz_py.save_xyz(f_name: str, labels: _Buffer | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes], coords: _Buffer | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes], with_numbers: bool = False, verbose: bool = True, mask: list = [], atomic_numbers: bool = False, comment: str = '')→ None
    save_path = get_unique_path(name, lambda i: f"{name}_f{i}.xyz", main_path)

    xyz_array = ADD
    #getting text answers about if we actually want an xyz file bc it needs comments on every line (the comments can just be empty strings)
    #also depending on the lbirary is xyz_array np.ndarray or jsut a regular array

    return NotImplementedError

def save_optical_flow_as_matlab(name: str, opt_flow: np.ndarray, main_path: str) -> None:
    """
    Saves the optical flow array as a MATLAB array file.

    Args:
        name (str): Name of the numpy array.
        opt_flow (np.ndarray): The optical flow array.
        main_path (str): Main path to the directory.
    
    Returns:
        None: Just saves the optical flow array to a file.
    """
    save_path = get_unique_path(name, lambda i: f"{name}_f{i}.mat", main_path)
    savemat(save_path, {"optical_flow": opt_flow})

def save_optical_flow_as_numpy(name: str, opt_flow: np.ndarray, main_path: str) -> None:
    """
    Saves the optical flow array as a numpy array.

    Args:
        name (str): Name of the numpy array.
        opt_flow (np.ndarray): The optical flow array.
        main_path (str): Main path to the directory.
    
    Returns:
        None: Just saves the optical flow array to a file.
    """
    save_path = get_unique_path(name, lambda i: f"{name}_f{i}.npy", main_path)
    np.save(save_path, opt_flow)

def save_original_video(name: str, file_path: str, im: matplotlib.image.AxesImage, image_stack: np.ndarray, fig: matplotlib.figure.Figure, ax: matplotlib.axes._axes.Axes, **kwargs) -> None:
    """
    Saves a video of image frames using matplotlib.

    Args:
        name (str): Name of the video file to save.
        file_path (str): The path to save the video file to.
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