import numpy as np
from multiprocessing import Pool, cpu_count
import cv2

def combine_flows(flow_list : list) -> np.ndarray:
    """
    Temporary function to combine different channels into one array.

    Args:
        flow_list (list[np.array]): List of numpy arrays to be combined.

    Returns:
        combined: combined stack of summed and original flows.
    """

    raise NotImplementedError

def compute_flow_pair(args) -> np.ndarray:
    """
    Computes optical flow for a pair of frames using Farneback method.

    Args:
        args (tuple): A tuple containing two frames and flow arguments.
            - f1: First frame (np.ndarray).
            - f2: Second frame (np.ndarray).
            - flow_args: Dictionary with parameters for optical flow calculation.
                - pyr_scale: float, scale factor for pyramid
                - levels: int, number of pyramid levels
                - winsize: int, size of the window for averaging
                - iterations: int, number of iterations at each pyramid level
                - poly_n: int, size of the pixel neighborhood
                - poly_sigma: float, standard deviation of the Gaussian used for polynomial expansion
                - flag: int, operation flags

    Returns:
        np.ndarray: Optical flow vectors for the pair of frames.
    """
    f1, f2, flow_args = args
    return cv2.calcOpticalFlowFarneback(
        f1, f2, None,
        flow_args['pyr_scale'],
        flow_args['levels'],
        flow_args['winsize'],
        flow_args['iterations'],
        flow_args['poly_n'],
        flow_args['poly_sigma'],
        flow_args['flag'])

def optical_flow(   arr : np.array,
                    pyr_scale : float = 0.5, 
                    levels : int = 3, 
                    winsize : int = 15,
                    iterations : int = 3, 
                    poly_n : int = 5, 
                    poly_sigma : float = 1.2,
                    flag : int = 0) -> np.ndarray:
    """
    Computes dense optical flow using Farneback method on a preprocessed channel. Allows manual
    changes to the params for optical flow.

    Args:
            - arr: np.arr, stack for optical flow processing
            - pyr_scale: float, scale factor for pyramid
            - levels: int, number of pyramid levels
            - winsize: int, size of the window for averaging
            - iterations: int, number of iterations at each pyramid level
            - poly_n: int, size of the pixel neighborhood
            - poly_sigma: float, standard deviation of the Gaussian used for polynomial expansion
            - flags: int, operation flags
            - **kwargs: dict with keys for preprocessing (see preprocess_frame)
                
    Returns:
        np.ndarray: (N-1, H, W, 2) flow vectors between frames.
    """ 
 
    flow_args = {
        'pyr_scale': pyr_scale,
        'levels': levels,
        'winsize': winsize,
        'iterations': iterations,
        'poly_n': poly_n,
        'poly_sigma': poly_sigma,
        'flag': flag
    }
    pairs = [(arr[i], arr[i+1], flow_args) for i in range(arr.shape[0] - 1)]
    with Pool(cpu_count()) as pool:
        flow_list = pool.map(compute_flow_pair, pairs)
    return np.stack(flow_list)

def calculate_optical_flow(self, process_args=None, flow_args=None, default=False):
        """
        Computes optical flow between the first two channels of the TIFF stack using the Farneback method.

        Args:
            process_args (dict): Preprocessing steps and parameters.
            flow_args (dict): Parameters for optical flow calculation.
            default (bool): Use default optical flow parameters if True.

        Returns:
            np.ndarray: Combined flow vectors of shape (N-1, H, W, 2).
        """
        raise NotImplementedError

def save_optflow_video(flow, idx : int = 0, step : int = 20, 
                          scale : int = 500, color : str = 'blue', fps : int = 10, 
                          figsize : int | int = (12,8),
                          title : str = None, overlay : bool = False):
     """
        Saves a video visualizing the optical flow.
     """
     raise NotImplementedError

def create_vector_field_video(name, arr : np.ndarray, og_arr : np.ndarray=None, 
                    step : int = 20, scale : int = 500, color : str = 'blue', 
                    fps : int = 10, figsize : int | int = (12,8),
                    title : str = None, flag : str = None) -> None:
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
    raise NotImplementedError

def calculate_trajectory(flow):
    """
    Calculates the trajectory of the optical flow vectors.

    Returns:
        np.ndarray: Trajectory of the optical flow vectors.
    """
    raise NotImplementedError
