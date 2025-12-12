import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.cell_tracking import optical_flow as of
from src.cell_tracking import raft
from src.cell_tracking import tiffclass
import numpy as np


def calculate_nuclei_optical_flow(
    arr: np.ndarray, channel: int, **kwargs
) -> np.ndarray:
    """
    This function calculates the optical flow for an isolated channel.

    Args:
      arr: The stack from the initialized tiff class

    Returns:
      The optical flow array.
    """
    return of.optical_flow(arr, channel, **kwargs)


def calculate_combined_flow(arr: np.ndarray, **kwargs) -> np.ndarray:
    """
    This function calculates the combined flow of channels 1 and 2.

    Args:
      arr: The stack from the initialized tiff class

    Returns:
      The optical flow array from combining the channels.
    """
    return of.calculate_optical_flow(arr, **kwargs)


def calculate_raft_optical_flow(Tiff: tiffclass.Tiff, **kwargs) -> np.ndarray:
    """
    This function calculates optical flow using the raft method.

    Args:
      arr: The stack from the initialized tiff class.

    Returns:
      The raft flow for the zeroth channel
    """
    return raft.calcOpticalFlowRAFT(Tiff, **kwargs)
