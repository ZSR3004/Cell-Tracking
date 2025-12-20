import os
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.cell_tracking import tiffclass
from . import file_input_cli as fic

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.cell_tracking import saving
import numpy as np


def save_flow_cli(name: str, opt_flow: np.ndarray, main_path: str):
    """
    Saves the raw data to the specified foler (raw data folder)

    Args:
        xyz_name: name of the xyz array
        matlab_name: name of the matlab array
        numpy_name: name of the numpy array
        opt_flow: the optical flow array to save
        main_path: the path to the file to save

    Returns:
        None: just saves the arrays to the specified directories
    """

    main_path = Path(main_path)

    saving.save_optical_flow_as_xyz(name, opt_flow, main_path)
    saving.save_optical_flow_as_matlab(name, opt_flow, main_path)
    saving.save_optical_flow_as_numpy(name, opt_flow, main_path)


def save_arr_cli(name: str, tiff_instance: tiffclass.Tiff, main_path: str) -> None:
    """
    Saves a numpy array to a file.

    Args:
        arr (np.array): The numpy array to save.

    Returns:
        None: Just saves the array to a file.
    """
    saving.save_arr(name, tiff_instance, main_path)


def save_original_video_cli(name: str, file_path: str, channel_idx: int) -> None:
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
    my_tiff = fic.init_tiff_class(file_path)

    image_stack = my_tiff.isolate_channel(channel_idx)

    output_dir = Path.cwd()
    output_path = output_dir / f"{name}.mp4"
    i = 1
    while output_path.exists():
        output_path = output_dir / f"{name}_{i}.mp4"
        i += 1

    fig, ax = plt.subplots()
    im = ax.imshow(image_stack[0], cmap="gray")
    ax.set_title("Frame" + str(channel_idx))

    saving.save_original_video(name, str(output_path), im, image_stack, fig, ax)
