import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import saving


def create_vector_field_video(name : str, arr : np.ndarray, og_arr : np.ndarray=None, 
                    step : int = 20, scale : int = 500, color : str = 'blue', 
                    fps : int = 10, figsize : int | int = (12,8),
                    title : str = None, flag : str = None) -> None:
    """
    Saves a video of optical flow (quiver animation), optionally overlaid on image frames.

    Args:
        name (str): Name of the video file to save.
        arr (np.ndarray): Optical flow array of shape (f-1, h, w, 2) where f-1 is the number of frames,
                          h is height, w is width, and the last dimension contains the flow vectors (dx, dy).
        og_arr (np.ndarray): Original image frames array of shape (f, c, h, w). Default is None.
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
        saving.save_vector_video(name, flag, 
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