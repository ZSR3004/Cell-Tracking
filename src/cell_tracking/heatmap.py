import numpy as np
import matplotlib.pyplot as plt

def convert_frame_to_polar(frame: np.ndarray) -> np.ndarray:
    raise NotImplementedError

def convert_stack_to_polar(frame_stack: np.ndarray) -> np.ndarray:
    raise NotImplementedError

def plot_heatmap(arr: np.ndarray, title: str, output_path: str, fps: int = 30) -> None:
    """
    Creates and saves a vector-magnitude heatmap to output_path.

    Args:
        arr (np.ndarray): The array to create a heatmap out of. This
            should be shape (f, c, h, w, 2) where 2 is (dx, dy) 
            which is in cartesian coordinates.
        title (str): Title for the heatmap.
        output_path (str): The path for the file to be saved to.
        fps (int): Fps of the output heatmap video.
    """
    raise NotImplementedError
