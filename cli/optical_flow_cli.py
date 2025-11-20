import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src import optical_flow as of
from src import raft
from src import tiffclass
import numpy as np

def calculate_nuclei_optical_flow(arr: np.ndarray, channel: int) -> np.ndarray:
    """
    This function preprocess the tiff stack with the parameters.

    Args:
      arr: The stack from the initialized tiff class
    
    Returns:
      The optical flow array. 
      """
    return of.optical_flow(arr, channel)

def calculate_combined_flow(arr: np.ndarray) -> np.ndarray:
    """
    This function preprocess the tiff stack with the parameters.

    Args:
      arr: The stack from the initialized tiff class
    
    Returns:
      The optical flow array from combining the channels. 
      """
    return of.calculate_optical_flow(arr)

def calculate_raft_optical_flow(Tiff: tiffclass.Tiff) -> np.ndarray:
    """
    This function preprocess the tiff stack with the parameters.

    Args:
      arr: The stack from the initialized tiff class.
    
    Returns:
      The raft flow for the third channel
      """
    return raft.calcOpticalFlowRAFT(Tiff)
