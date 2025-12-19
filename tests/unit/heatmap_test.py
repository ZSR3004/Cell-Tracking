import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2, pytest, tifffile
from src.cell_tracking import heatmap
from src.cell_tracking import tiffclass as tiff
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from unittest.mock import patch, Mock, MagicMock
from tests.unit.sample_tiffs import TIFF_PATHS


@pytest.fixture(params=TIFF_PATHS)
def init_tiff(request: pytest.FixtureRequest):
    """
    Creates a Tiff class instance.

    Args:
        request (pytest.FixtureRequest): The paths to generate Tiff instances from.

    Returns:
        (tiff.Tiff): A tuple containing information about the TIFF file.
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
    """
    path = request.param
    img = tifffile.imread(path)
    info = (img.shape[0], img.shape[1], img.shape[2], img.shape[3])
    return (tiff.Tiff(path), info)


def test_convert_stack_to_polar(init_tiff: tuple):
    """
    Tests whether the convert_stack_to_polar function works correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file.
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Returns:
        None.
    """
    img, info = init_tiff
    f, c, h, w = info
    tiff_arr = img.arr

    def dummy_optical_flow(channel: int):
        """
        A function that creates an array of shape (f-1, h, w, 2).

        Args:
            channel (int): The channel to process.

        Returns:
            A np.ndarray of shape (f-1, h, w, 2).
        """
        arr_channel = tiff_arr[:, channel, :, :]

        dummy_dx = arr_channel[1:] - arr_channel[:-1]
        dummy_dy = arr_channel[1:] - arr_channel[:-1]

        return np.stack([dummy_dx, dummy_dy], axis=-1)

    # flow0, flow1, and flow2 have shape (f-1, h, w, 2)
    flow0 = dummy_optical_flow(0)
    flow1 = dummy_optical_flow(1)
    flow2 = dummy_optical_flow(2)

    def test_case_x(flowx: np.ndarray):
        """
        Tests whether the convert_stack_to_polar function works correctly on a specific test case.

        Args:
            flowx (np.ndarray): The numpy array representation
            of the TIFF file. Shape is (f-1, h, w, 2), where 2 is (dx, dy).

        Returns:
            None.
        """
        x = flowx[..., 0]
        y = flowx[..., 1]
        r = np.sqrt(x**2 + y**2)
        max_f = np.max(np.abs(flowx))
        r_norm = r / (np.sqrt(2) * max_f)
        theta = np.arctan2(y, x)
        expected_result = np.stack([r_norm, theta], axis=-1)

        result = heatmap.convert_stack_to_polar(flowx)

        assert result.shape == (f-1, h, w, 2)
        assert result.shape == flowx.shape
        assert result.shape == expected_result.shape
        assert np.array_equal(result, expected_result)

    test_case_x(flow0)
    test_case_x(flow1)
    test_case_x(flow2)


def test_create_color_wheel():
    """
    Cont when I get the answers to these questions:
    What is supposed to be in the returns part of the docstring?
    """


def test_polar_to_heatmap():
    """
    Cont when I get the answers to these questions:
    When it says "last dim is (r, theta)", r and theta are just numbers, right? (If so, when I'm creating the dummy array for testing it doesnt matter what
    the numbers r and theta are) YES
    """
    pass


def test_plot_heatmap():
    pass


# note: when working with stuff of shape like (f, c, h, w, 2), use stuff i wrote for test_plot_basic_kymo
