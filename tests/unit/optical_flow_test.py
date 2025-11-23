import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import sys, os, pytest
from sympy import Idx
from src.cell_tracking import optical_flow as flow
import numpy as np
from src.cell_tracking import tiffclass as tiff
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
import cv2

TIFF_PATHS = [
    (
        "datasets/nuclei_labeled/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif",
        (96, 3, 520, 2329),
    )
]


@pytest.fixture(params=TIFF_PATHS)
def init_tiff(request: pytest.FixtureRequest) -> tiff.Tiff:
    """
    Creates a Tiff class instance.

    Args:
        request (pytest.FixtureRequest): The paths to generate Tiff instances from.

    Returns:
        (tiff.Tiff): Tiff class instance of the path.
    """
    path, info = request.param
    return (tiff.Tiff(path), info)


def test_combine_flows(sample_tiff):
    """
    Tests whether the combine_flows function works correctly.

    Args:
        sample_tiff (tuple): A tuple containing information about the TIFF file.
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Returns:
        None.
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)

    flow_0 = flow.optical_flow(img.arr, 0)
    flow_1 = flow.optical_flow(img.arr, 1)
    flow_2 = flow.optical_flow(img.arr, 2)

    combine_flow_0 = flow.combine_flows([flow_0, flow_1])
    combine_flow_1 = flow.combine_flows([flow_1, flow_2])
    combine_flow_2 = flow.combine_flows([flow_0, flow_2])

    assert combine_flow_0.shape == (f - 1, c, h, w, 2)
    assert combine_flow_1.shape == (f - 1, c, h, w, 2)
    assert combine_flow_2.shape == (f - 1, c, h, w, 2)


def test_compute_flow_pair(sample_tiff):
    """
    Tests whether the compute_flow_pair function works correctly.

    Args:
        sample_tiff (tuple): A tuple containing information about the TIFF file.
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Returns:
        None.
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)

    # Grab the first two frames of channel 0
    f1 = img.arr[0, 0]  # shape: (height, width)
    f2 = img.arr[1, 0]  # shape: (height, width)

    flow_args = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }

    args = (f1, f2, flow_args)
    my_flow = flow.compute_flow_pair(args)

    # Test output type
    assert isinstance(my_flow, np.ndarray)

    # Test output shape
    assert my_flow.shape == (h, w, 2)


def test_optical_flow(sample_tiff):
    """
    Tests whether the optical_flow function works correctly.

    Args:
        sample_tiff (tuple): A tuple containing information about the TIFF file.
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Returns:
        None.
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)

    # arguments for testing
    flow_args1 = {
        "pyr_scale": 0.75,
        "levels": 5,
        "winsize": 17,
        "iterations": 5,
        "poly_n": 10,
        "poly_sigma": 1.4,
        "flags": 1,
    }
    flow_args2 = {"levels": 5, "winsize": 17, "poly_n": 10, "flags": 1}
    flow_args3 = {}

    flow1_channel0 = flow.optical_flow(img.arr, 0, **flow_args1)
    flow1_channel1 = flow.optical_flow(img.arr, 1, **flow_args1)
    flow1_channel2 = flow.optical_flow(img.arr, 2, **flow_args1)
    flow2_channel0 = flow.optical_flow(img.arr, 0, **flow_args2)
    flow2_channel1 = flow.optical_flow(img.arr, 1, **flow_args2)
    flow2_channel2 = flow.optical_flow(img.arr, 2, **flow_args2)
    flow3_channel0 = flow.optical_flow(img.arr, 0, **flow_args3)
    flow3_channel1 = flow.optical_flow(img.arr, 1, **flow_args3)
    flow3_channel2 = flow.optical_flow(img.arr, 2, **flow_args3)

    # Test output type
    assert isinstance(flow1_channel0, np.ndarray)
    assert isinstance(flow1_channel1, np.ndarray)
    assert isinstance(flow1_channel2, np.ndarray)
    assert isinstance(flow2_channel0, np.ndarray)
    assert isinstance(flow2_channel1, np.ndarray)
    assert isinstance(flow2_channel2, np.ndarray)
    assert isinstance(flow3_channel0, np.ndarray)
    assert isinstance(flow3_channel1, np.ndarray)
    assert isinstance(flow3_channel2, np.ndarray)

    # Test output shape
    assert flow1_channel0.shape == (f - 1, h, w, 2)
    assert flow1_channel1.shape == (f - 1, h, w, 2)
    assert flow1_channel2.shape == (f - 1, h, w, 2)
    assert flow2_channel0.shape == (f - 1, h, w, 2)
    assert flow2_channel1.shape == (f - 1, h, w, 2)
    assert flow2_channel2.shape == (f - 1, h, w, 2)
    assert flow3_channel0.shape == (f - 1, h, w, 2)
    assert flow3_channel1.shape == (f - 1, h, w, 2)
    assert flow3_channel2.shape == (f - 1, h, w, 2)

    # Test if output is not an array with only zeros
    all_zeros = np.zeros((f - 1, h, w, 2), order="C")

    assert not np.allclose(all_zeros, flow1_channel0)
    assert not np.allclose(all_zeros, flow1_channel1)
    assert not np.allclose(all_zeros, flow1_channel2)
    assert not np.allclose(all_zeros, flow2_channel0)
    assert not np.allclose(all_zeros, flow2_channel1)
    assert not np.allclose(all_zeros, flow2_channel2)
    assert not np.allclose(all_zeros, flow3_channel0)
    assert not np.allclose(all_zeros, flow3_channel1)
    assert not np.allclose(all_zeros, flow3_channel2)


def test_calculate_optical_flow(sample_tiff):
    """
    Tests whether the calculate_optical_flow function works correctly.

    Args:
        sample_tiff (tuple): A tuple containing information about the TIFF file.
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Returns:
        None.
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)

    # Example preprocessing: normalize frames to 0-1, apply small Gaussian blur
    process_args = {
        "normalize": {"alpha": 0, "beta": 255, "norm_type": cv2.NORM_MINMAX},
        "gauss": {"ksize": (5, 5), "sigmaX": 1.5},
    }

    my_flow = flow.calculate_optical_flow(img.arr, **process_args)

    # Test output type
    assert isinstance(my_flow, np.ndarray)

    # Test output shape
    assert my_flow.shape == (f - 1, c, h, w, 2)


def test_show_flow(sample_tiff):
    """
    Tests the show_flow function.

    Args:
        sample_tiff (tuple): A tuple containing information about the TIFF file.
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Returns:
        None.
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)

    my_flow = flow.optical_flow(img.arr, 0)
    first_flow_frame = my_flow[0]
    video = flow.show_flow(
        first_flow_frame, "Optical Flow", 25, (12, 6), 200, "tail", "blue", None
    )

    fig = plt.gcf()
    assert isinstance(fig, Figure)
    plt.close()
