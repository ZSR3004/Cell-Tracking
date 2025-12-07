import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def flatten_arr(arr: np.ndarray):
    """
    Flattens the input array along the last axis and returns a 2D array.
    The input array is expected to have a shape of (frames-1, height, width, 2),
    where the last dimension contains the flow vectors (dx, dy).

    Args:
        arr (np.ndarray): Input array of shape (frames-1, height, width, 2).

    Returns:
        np.ndarray: Flattened 2D array of shape (frames-1, width).
    """
    mag_per_frame = np.linalg.norm(arr, axis=-1)
    mag_arr = np.array([np.median(mag_per_frame[i, :, :], axis=0) for i in range(arr.shape[0])])
    return mag_arr

def mask_line_arr(line_arr: np.ndarray, threshold: int=0.5):
    """
    Masks the input array by setting values below a threshold to zero and
    keeping the maximum value in the array. This is useful for visualizing
    flow vectors in a kymograph.

    Args:
        line_arr (np.ndarray): Input array of shape (frames, height, width).
        threshold (int, optional): Threshold value to mask the array. Defaults to 0.5.

    Returns:
        np.ndarray: Masked array where values below the threshold are set to zero, and the 
                    maximum value in the array is retained. Shape is (frames, height, width).
    """
    max_val = np.max(line_arr)
    masked_line_arr = np.where(line_arr > threshold, max_val, 0)
    return masked_line_arr

def plot_basic_kymo(arr: np.ndarray, save_path: str=None, threshold: float=0.5):
    """
    Plots a kymograph from the input array, which is expected to be a 4D array
    with shape (frames-1, channel, height, width, 2). The kymograph visualizes
    flow vectors over time, with different colors representing different
    flow directions.

    Args:
        arr (np.ndarray): Input array of shape (frames-1, channel, height, width, 2), where the last dimension contains the flow vectors (dx, dy).
        save_path (str): Path to save the plot. If None, the plot will be displayed instead of saved.
        threshold (float, optional): Threshold value to mask the array. Defaults to 0.5.
    
    Returns:
        None: Displays or saves the kymograph plot.
              The plot shows the flow vectors over time, with different colors representing
              different flow directions. The left flow is shown in green, the right flow
              in magenta, and the overlap in blue.
              The plot is saved to the specified path or displayed if no path is provided.
    """
    def mask_boundary(channel_arr: np.ndarray, threshold: float=0.5):
        return mask_line_arr(flatten_arr(channel_arr))

    masked_line_arr1 = mask_boundary(arr[:, 1, ...], threshold=threshold)
    masked_line_arr2 = mask_boundary(arr[:, 2, ...], threshold=threshold)

    combined_data = np.zeros_like(masked_line_arr1)
    combined_data[masked_line_arr1 != 0] += 1
    combined_data[masked_line_arr2 != 0] += 2

    custom_green = (119/255, 237/255, 130/255)
    custom_magenta = (201/255, 107/255, 232/255)
    custom_blue = (18/255, 105/255, 204/255)
    colors = ['black', custom_green, custom_magenta, custom_blue]
    cmap = mcolors.ListedColormap(colors)

    plt.figure(figsize=(10, 5))
    plt.imshow(combined_data, aspect='auto', cmap=cmap, vmin=0, vmax=3)
    plt.title("Overlay: Left (Green), Right (Magenta), Overlap (Blue)")
    plt.xlabel('Position')
    plt.ylabel('Time')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()