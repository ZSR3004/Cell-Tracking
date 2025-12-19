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


def test_convert_stack_to_polar():
    pass


def test_create_color_wheel():
    """
    Cont when I get the answers to these questions:
    What is supposed to be in the returns part of the docstring?
    """
    pass


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
