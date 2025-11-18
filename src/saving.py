import os
import json
import numpy as np
import pathlib
from pathlib import Path
import matplotlib.animation as animation
import scipy.io 
from src import tiffclass as tiff
from src.defaults import default_process, default_flow, default_trajectory
import matplotlib.image, matplotlib.figure, matplotlib.axes


def get_unique_path(name: str, file_type: str, pattern_fn, main_path: str) -> Path:
    """
    Generates a unique file path in the given directory based on a naming pattern.

    Args:
        name (str): Main identifier (e.g., protein name).
        file_type (str): Subdirectory (e.g., 'flow', 'trajectory').
        pattern_fn (callable): Function that takes an integer and returns a file name.
        main_path (str): Main path to the directory.

    Returns:
        Path: Unique file path that does not yet exist.
    """
    save_dir = main_path / name / file_type
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
    np.save(main_path / name / 'arr', tiff_arr)

def save_optical_flow_as_xyz(opt_flow: np.ndarray, save_path: str) -> None:
    """
    Saves the optical flow array as an XYZ file. Saves it to the input path (which is the path to a folder).

    Args:
        opt_flow (np.ndarray): The optical flow array.
        save_path (str): The path to the folder where the output (an XYZ array) will be saved.
    
    Returns:
        None: Just saves the optical flow array to a file.
    """
    #Caroline
    #turn optical flow array into xyz file and then save it to the folder
    #first figure out the shape of the optical flow array. Ziyad says (frames, channel, height, width, 2) 
    #the 2 is a (dx, dy) tuple. You use a library called atomic xyz pipeline. CASEY IS TEXTING ABOUT IT
    #you need to save it to the hard drive. Do it using np.save()
    return NotImplementedError

def save_optical_flow_as_matlab(opt_flow: np.ndarray, save_path: str) -> None:
    """
    Saves the optical flow array as a MATLAB array. Saves it to the input path (which is the path to a folder).

    Args:
        opt_flow (np.ndarray): The optical flow array.
        save_path (str): The path to the folder where the output (an XYZ array) will be saved.
    
    Returns:
        None: Just saves the optical flow array to a file.
    """
    #Caroline
    #turn optical flow array into matlab array and then save it to the folder
    #first figure out the shape of the optical flow array. Ziyad says (frames, channel, height, width, 2) 
    #use scipy.io.savemat
    #you need to save it to the hard drive. Do it using np.save()
    return NotImplementedError

def save_optical_flow_as_numpy(opt_flow: np.ndarray, save_path: str) -> None:
    """
    Saves the optical flow array as a numpy array. Saves it to the input path (which is the path to a folder).

    Args:
        opt_flow (np.ndarray): The optical flow array.
        save_path (str): The path to the folder where the output (an XYZ array) will be saved.
    
    Returns:
        None: Just saves the optical flow array to a file.
    """
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