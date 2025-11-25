import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2, pytest
from sympy import Idx
from src.cell_tracking import optical_flow as flow
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from multiprocessing import Pool, cpu_count
from src.cell_tracking import tiffclass as tiff
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver

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


def test_combine_flows(init_tiff):
    """
    Tests whether the combine_flows function works correctly.

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

    flow_0 = flow.optical_flow(tiff_arr, 0)
    flow_1 = flow.optical_flow(tiff_arr, 1)
    flow_2 = flow.optical_flow(tiff_arr, 2)

    combine_flow_0 = flow.combine_flows([flow_0, flow_1])
    combine_flow_1 = flow.combine_flows([flow_1, flow_2])
    combine_flow_2 = flow.combine_flows([flow_0, flow_2])

    assert combine_flow_0.shape == (f - 1, c, h, w, 2)
    assert combine_flow_1.shape == (f - 1, c, h, w, 2)
    assert combine_flow_2.shape == (f - 1, c, h, w, 2)


def test_compute_flow_pair(init_tiff):
    """
    Tests whether the compute_flow_pair function works correctly.

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

    # shapes of these: (height, width)
    frame0_channel0 = tiff_arr[0, 0]  
    frame1_channel0 = tiff_arr[1, 0]
    middleframe0_channel1 = tiff_arr[f//2, 1]
    middleframe1_channel1 = tiff_arr[(f//2)-1, 1]
    lastframe0_channel2 = tiff_arr[-1, 2]
    lastframe1_channel2 = tiff_arr[-2, 2]

    flow_args1 = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }
    flow_args2 = {
        "pyr_scale": 0.75,
        "levels": 5,
        "winsize": 17,
        "iterations": 6,
        "poly_n": 3,
        "poly_sigma": 1.8,
        "flags": 1,
    }

    def test_case_x(f1: np.ndarray, f2: np.ndarray, flow_argsx: dict):
        """
        Tests whether the compute_flow_pair function works correctly on a specific test case.

        Args:
            f1 (np.ndarray): First frame.
            f2 (np.ndarray): Second frame.
            flow_argsx (dict): Dictionary with parameters for optical flow calculation.
                - pyr_scale (float): Scale factor for pyramid.
                - levels (int): Number of pyramid levels.
                - winsize (int): Size of the window for averaging.
                - iterations (int): Number of iterations at each pyramid level.
                - poly_n (int): Size of the pixel neighborhood.
                - poly_sigma (float): Standard deviation of the Gaussian used for polynomial expansion.
                - flag (int): Operation flags

        Returns:
            None.
        """
        with patch("cv2.calcOpticalFlowFarneback") as mock_farneback:
            fake_flow = np.zeros((h, w, 2), dtype=f1.dtype)
            mock_farneback.return_value = fake_flow

            my_flow = flow.compute_flow_pair((f1, f2, flow_argsx))

            farneback_args, _ = mock_farneback.call_args
            assert np.array_equal(farneback_args[0], f1)
            assert np.array_equal(farneback_args[1], f2)
            assert farneback_args[2] == None
            assert farneback_args[3] == flow_argsx["pyr_scale"]
            assert farneback_args[4] == flow_argsx["levels"]
            assert farneback_args[5] == flow_argsx["winsize"]
            assert farneback_args[6] == flow_argsx["iterations"]
            assert farneback_args[7] == flow_argsx["poly_n"]
            assert farneback_args[8] == flow_argsx["poly_sigma"]
            assert farneback_args[9] == flow_argsx["flags"]

            assert my_flow.shape == fake_flow.shape
            assert my_flow.shape == (h, w, 2)
            assert np.array_equal(my_flow, fake_flow)
            assert isinstance(my_flow, np.ndarray)
            mock_farneback.assert_called_once()

    test_case_x(frame0_channel0, frame1_channel0, flow_args1)
    test_case_x(frame0_channel0, frame1_channel0, flow_args2)
    test_case_x(middleframe0_channel1, middleframe1_channel1, flow_args1)
    test_case_x(middleframe0_channel1, middleframe1_channel1, flow_args2)
    test_case_x(lastframe0_channel2, lastframe1_channel2, flow_args1)
    test_case_x(lastframe0_channel2, lastframe1_channel2, flow_args2)


def test_optical_flow(init_tiff):
    """
    Tests whether the optical_flow function works correctly.

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

    kwargs1 = {
        "pyr_scale": 0.75,
        "levels": 5,
        "winsize": 17,
        "iterations": 5,
        "poly_n": 10,
        "poly_sigma": 1.4,
        "flags": 1,
    }
    kwargs2 = {"levels": 5, "winsize": 17, "poly_n": 10, "flags": 1}
    kwargs3 = {}

    def test_case_x(channelx: int, **kwargsx):
        """
        Computes dense optical flow using Farneback method on a preprocessed channel.
        Accepts all Farneback parameters as keyword arguments.

        Args:
            channelx (int): The channel to process.
            **kwargsx: Additional keyword arguments passed to cv2.calcOpticalFlowFarneback:
                - pyr_scale (float): Scale factor for pyramid. Default 0.5
                - levels (int): Number of pyramid levels. Default 3
                - winsize (int): Window size for averaging. Default 15
                - iterations (int): Number of iterations per pyramid level. Default 3
                - poly_n (int): Size of pixel neighborhood. Default 5
                - poly_sigma (float): Gaussian std for polynomial expansion. Default 1.2
                - flags (int): Operation flags. Default 0

        Returns:
            None.
        """
        with patch("src.cell_tracking.optical_flow.Pool") as mock_pool:
            mock_pool_instance = mock_pool.return_value.__enter__.return_value
            mock_pool_instance.map.side_effect = lambda func, arr1: [np.zeros((len(arr1[0][0]), len(arr1[0][1]), 2)) for i in range len(arr1)]

            flow_args = {
                "pyr_scale": 0.5,
                "levels": 3,
                "winsize": 15,
                "iterations": 3,
                "poly_n": 5,
                "poly_sigma": 1.2,
                "flags": 0,
            }
            flow_args.update(kwargsx)
            arr_channel = tiff_arr[:, channelx, :, :]
            pairs = [
                (arr_channel[i], arr_channel[i + 1], flow_args)
                for i in range(arr_channel.shape[0] - 1)
            ]

            result = flow.optical_flow(tiff_arr, channelx, **kwargsx)

            pool_args, _ = mock_pool.call_args
            assert pool_args[0] == cpu_count()

            pool_map_args, _ = mock_pool_instance.map.call_args
            assert callable(pool_map_args[0])
            for i in range(0, len(pairs)):
                assert np.array_equal(pool_map_args[1][i][0], pairs[i][0])
                assert np.array_equal(pool_map_args[1][i][1], pairs[i][1])
                assert pool_map_args[1][i][2] == pairs[i][2]

            assert isinstance(result, np.ndarray)
            mock_pool.assert_called_once()
            mock_pool_instance.map.assert_called_once()

            assert result.shape == (f-1, h, w, 2)

    test_case_x(0, **kwargs1)
    test_case_x(1, **kwargs1)
    test_case_x(2, **kwargs1)
    test_case_x(0, **kwargs2)
    test_case_x(1, **kwargs2)
    test_case_x(2, **kwargs2)
    test_case_x(0, **kwargs3)
    test_case_x(1, **kwargs3)
    test_case_x(2, **kwargs3)


def test_calculate_optical_flow(init_tiff):
    """
    Tests whether the calculate_optical_flow function works correctly.

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

    # Example preprocessing: normalize frames to 0-1, apply small Gaussian blur
    process_args = {
        "normalize": {"alpha": 0, "beta": 255, "norm_type": cv2.NORM_MINMAX},
        "gauss": {"ksize": (5, 5), "sigmaX": 1.5},
    }

    my_flow = flow.calculate_optical_flow(tiff_arr, **process_args)

    # Test output type
    assert isinstance(my_flow, np.ndarray)

    # Test output shape
    assert my_flow.shape == (f - 1, c, h, w, 2)

    #MOCK AND PATCH
    return NotImplementedError


def test_show_flow(init_tiff):
    """
    Tests the show_flow function.

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

    my_flow = flow.optical_flow(tiff_arr, 0)
    first_flow_frame = my_flow[0]
    video = flow.show_flow(
        first_flow_frame, "Optical Flow", 25, (12, 6), 200, "tail", "blue", None
    )

    fig = plt.gcf()
    assert isinstance(fig, Figure)
    plt.close()

    #MOCK AND PATCH
    return NotImplementedError