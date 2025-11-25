import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from src.cell_tracking import tiffclass as tc
from src.cell_tracking import saving as save


def combine_flows(flow_list: list) -> np.ndarray:
    """
    Temporary function to combine different channels into one array.

    Args:
        flow_list (list[np.ndarray]): List of numpy arrays to be combined.

    Returns:
        combined (np.ndarray): Combined stack of summed and original flows.
    """
    sum_arr = flow_list[0] + flow_list[1]
    combined = np.stack([sum_arr, flow_list[0], flow_list[1]], axis=1)
    return combined


def compute_flow_pair(args) -> np.ndarray:
    """
    Computes optical flow for a pair of frames using Farneback method.

    Args:
        args (tuple): A tuple containing two frames and flow arguments.
            - f1 (np.ndarray): First frame.
            - f2 (np.ndarray): Second frame.
            - flow_args (dict): Dictionary with parameters for optical flow calculation.
                - pyr_scale (float): Scale factor for pyramid.
                - levels (int): Number of pyramid levels.
                - winsize (int): Size of the window for averaging.
                - iterations (int): Number of iterations at each pyramid level.
                - poly_n (int): Size of the pixel neighborhood.
                - poly_sigma (float): Standard deviation of the Gaussian used for polynomial expansion.
                - flag (int): Operation flags

    Returns:
        np.ndarray: Optical flow vectors for the pair of frames.
    """
    f1, f2, flow_args = args
    return cv2.calcOpticalFlowFarneback(
        f1,
        f2,
        None,
        flow_args["pyr_scale"],
        flow_args["levels"],
        flow_args["winsize"],
        flow_args["iterations"],
        flow_args["poly_n"],
        flow_args["poly_sigma"],
        flow_args["flags"],
    )


def optical_flow(arr: np.ndarray, channel: int, **kwargs) -> np.ndarray:
    """
    Computes dense optical flow using Farneback method on a preprocessed channel.
    Accepts all Farneback parameters as keyword arguments.

    Args:
        arr (np.ndarray): Stack for optical flow processing.
        channel (int): The channel to process.
        **kwargs: Additional keyword arguments passed to cv2.calcOpticalFlowFarneback:
            - pyr_scale (float): Scale factor for pyramid. Default 0.5
            - levels (int): Number of pyramid levels. Default 3
            - winsize (int): Window size for averaging. Default 15
            - iterations (int): Number of iterations per pyramid level. Default 3
            - poly_n (int): Size of pixel neighborhood. Default 5
            - poly_sigma (float): Gaussian std for polynomial expansion. Default 1.2
            - flags (int): Operation flags. Default 0

    Returns:
        np.ndarray: Flow vectors of shape (N-1, H, W, 2) between frames.
    """
    flow_args = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }

    flow_args.update(kwargs)

    arr_channel = arr[:, channel, :, :]
    pairs = [
        (arr_channel[i], arr_channel[i + 1], flow_args)
        for i in range(arr_channel.shape[0] - 1)
    ]

    with Pool(cpu_count()) as pool:
        flow_list = pool.map(compute_flow_pair, pairs)

    return np.stack(flow_list)


def calculate_optical_flow(arr: np.ndarray, **kwargs) -> np.ndarray:
    """
    Computes optical flow between the first two channels of the TIFF stack using the Farneback method.
    Accepts any Farneback parameters as keyword arguments.

    Args:
        arr (np.ndarray): The TIFF stack array to process.
        **kwargs: Additional keyword arguments passed to `optical_flow` for Farneback parameters:
            - pyr_scale (float)
            - levels (int)
            - winsize (int)
            - iterations (int)
            - poly_n (int)
            - poly_sigma (float)
            - flags (int)

    Returns:
        np.ndarray: Combined flow vectors of shape (N-1, 3, H, W, 2).
    """
    flow_channel1 = optical_flow(arr, 1, **kwargs)
    flow_channel2 = optical_flow(arr, 2, **kwargs)
    combined = combine_flows([flow_channel1, flow_channel2])
    return combined


def show_flow(
    flow: np.ndarray,
    title="Optical Flow",
    step: int = 25,
    figsize: int | int = (12, 6),
    scale: int = 200,
    pivot: str = "tail",
    color: str = "blue",
    save_path: str = None,
) -> None:
    """
    Displays optical flow as a quiver plot using matplotlib.

    Args:
        flow (np.ndarray): Optical flow array of shape (H, W, 2) where H is height, W is width,
                           and the last dimension contains the flow vectors (dx, dy).
        title (str): Title of the plot. Default is 'Optical Flow'.
        step (int): Step size for downsampling the flow vectors for visualization. Default is 25.
        figsize (tuple): Size of the figure in inches (width, height). Default is (12, 6).
        scale (float): Scale factor for the quiver arrows. Default is 200.
        pivot (str): Pivot point for the arrows. Default is 'tail'.
        color (str): Color of the arrows. Default is 'white'.

    Returns:
        None: Just displays the plot.
    """
    Y, X = np.mgrid[0 : flow.shape[0] : step, 0 : flow.shape[1] : step]
    U = flow[::step, ::step, 0]  # dx
    V = flow[::step, ::step, 1]  # dy

    # Create plot
    plt.figure(figsize=figsize)
    plt.quiver(X, Y, U, V, scale=scale, pivot=pivot, color=color)
    plt.title(title)
    plt.xlim(0, flow.shape[1])
    plt.ylim(flow.shape[0], 0)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
