import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.cell_tracking import tiffclass as tiff
import numpy as np


def init_tiff_class(path: str) -> tiff.Tiff:
    """
    This function initializes a tiff class from the command line.

    Args:
      The path name to a tiff file.

    Assumptions:
      Assumes input leads to a valid tiff file.
    
    Returns:
      A tiff class.
    """
    my_class = tiff.Tiff(path)
    return my_class

def preprocess_tiff(tiff_obj: tiff.Tiff, **kwargs) -> np.ndarray:
    """
    This function preprocess the tiff stack with the parameters.

    Args:
      tiff = a Tiff class
      **kwargs: Dictionary with preprocessing parameters:
            - gauss (dict): {'ksize': (int, int), 'sigmaX': float}
            - median (dict): {'ksize': int}
            - normalize (dict): {'alpha': int, 'beta': int, 'norm_type': int}
            - contrast (dict): {'alpha': float, 'beta': int}
            - skip (list[str]): steps to skip (e.g., ['gauss', 'median'])
    
    Returns:
      The preprocessed stack (as a numpy array.)
    """
    return tiff_obj.preprocess_stack(tiff_obj.arr, **kwargs)
