import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace
from multiprocessing import Pool, cpu_count
import matplotlib as plt
from src import tiffclass as tc
from src import saving as save

def combine_flows(flow_list : list) -> np.ndarray:
    """
    Temporary function to combine different channels into one array.

    Args:
        flow_list (list[np.array]): List of numpy arrays to be combined.

    Returns:
        combined: combined stack of summed and original flows.
    """

    sum_arr = flow_list[0] + flow_list[1]
    combined = np.stack([sum_arr, flow_list[0], flow_list[1]], axis=1)
    return combined


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

def calculate_optical_flow(arr: np.array, process_args=None, default=False):
        """
        Computes optical flow between the first two channels of the TIFF stack using the Farneback method.

        Args:
            process_args (dict): Preprocessing steps and parameters.
            flow_args (dict): Parameters for optical flow calculation.
            default (bool): Use default optical flow parameters if True.

        Returns:
            np.ndarray: Combined flow vectors of shape (N-1, H, W, 2).
        """
        return optical_flow(tc.preprocess_stack(arr, process_args), 0.5, 3, 15, 3, 5, 1.2,0)

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
    T, H, W, _ = arr.shape
    Y, X = np.mgrid[0:H:step, 0:W:step]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Optical Flow")
    ax.set_aspect('equal')
    ax.axis('off')

    if og_arr is not None:
        is_gray = og_arr.ndim == 3
        img_disp = ax.imshow(og_arr[0], cmap='gray' if is_gray else None)
    else:
        img_disp = None

    U = arr[0, ::step, ::step, 0]
    V = arr[0, ::step, ::step, 1]
    quiver = ax.quiver(X, Y, U, V, scale=scale, pivot='tail', color=color)

    if flag != "":
        save.save_vector_video(name, flag, 
                   **{
                          'img_disp': img_disp,
                          'arr': arr,
                          'og_arr': og_arr,
                          'step': step,
                          'fps': fps,
                          'figsize': figsize,
                          'title': title,
                          'quiver': quiver,
                          'ax': ax,
                          'fig': fig,
                          'T': T
                   })

    plt.close(fig)

def save_optflow_video(name, flow, save_path, idx : int = 0, step : int = 20, 
                                  scale : int = 500, color : str = 'blue', fps : int = 10, 
                                  figsize : int | int = (12,8),
                                  title : str = None, overlay : bool = False):
    """
        Saves a video visualizing the optical flow.
    """
    if overlay:
        og_arr = tc.isolate_channel(idx)
    else:
        og_arr = None 

        create_vector_field_video(
            name, 
            flow[:, idx, ...], 
            og_arr, 
            step=step, 
            scale=scale,
            color=color, 
            fps=fps, 
            figsize=figsize, 
            title=title,
            flag='f'
        )
    

def show_flow(flow : np.array, title='Optical Flow', 
              step : int = 25, figsize : int | int = (12,6), scale : int = 200, 
              pivot : str = 'tail', color : str = 'blue', save_path : str = None) -> None:
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
    Y, X = np.mgrid[0:flow.shape[0]:step, 0:flow.shape[1]:step]
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
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
