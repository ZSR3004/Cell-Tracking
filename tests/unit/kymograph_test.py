import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock, call
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.cell_tracking import tiffclass as tiff
from src.cell_tracking import kymograph

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


def test_flatten_arr(init_tiff: tuple):
    """
    Tests whether the flatten_arr function works correctly.

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
        A function that creates an array similar to (and with the same shape as) flow.optical_flow(tiff_arr, channel). This function is much
        faster than calling flow.optical_flow (which is why it's perfect for testing).

        Args:
            channel (int): The channel to process.

        Returns:
            A np.ndarray of shape (f-1, h, w, 2).
        """
        arr_channel = tiff_arr[:, channel, :, :]

        dummy_dx = arr_channel[1:] - arr_channel[:-1]
        dummy_dy = arr_channel[1:] - arr_channel[:-1]

        return np.stack([dummy_dx, dummy_dy], axis=-1)

    #flow0, flow1, and flow2 have shape (f-1, h, w, 2), which is the same shape as flow.optical_flow(tiff_arr, n) (where n is 0, 1, or 2)
    flow0 = dummy_optical_flow(0)
    flow1 = dummy_optical_flow(1)
    flow2 = dummy_optical_flow(2)

    def test_case_x(flowx: np.ndarray):
        """
        Tests whether the vector_magnitude_heatmaps function works correctly on a specific test case.

        Args:
            flowx (np.ndarray): Flow array of shape (frames-1, height, width, 2).

        Returns:
            None.
        """
        with patch("numpy.linalg.norm") as mock_linalg_norm, \
            patch("numpy.median") as mock_median:
            #fake_flow has shape (f-1, h, w), which is the same shape as np.linalg.norm(flowx, axis=-1)
            fake_flow = flowx[..., 0]
            mock_linalg_norm.return_value = fake_flow
            #mock_median.side_effect returns an array of shape (w), which is the same shape as np.median(mag_per_frame[i, :, :], axis=0) (note: the shape of mag_per_frame[i, :, :] is (h, w))
            mock_median.side_effect = lambda arr1, **axis1: arr1[0, :]

            result = kymograph.flatten_arr(flowx)

            args_linalg_norm, kwargs_linalg_norm = mock_linalg_norm.call_args
            assert np.array_equal(args_linalg_norm[0], flowx)
            assert kwargs_linalg_norm["axis"] == -1

            for i, call_args_kwargs in enumerate(mock_median.call_args_list):
                args_median, kwargs_median = call_args_kwargs
                assert np.array_equal(args_median[0], fake_flow[i, :, :])
                assert kwargs_median["axis"] == 0

            mock_linalg_norm.assert_called_once()
            assert mock_median.call_count == flowx.shape[0]

            assert result.shape == (f-1, w)
            assert isinstance(result, np.ndarray)

    test_case_x(flow0)
    test_case_x(flow1)
    test_case_x(flow2)


def test_mask_line_arr(init_tiff: tuple):
    """
    Tests whether the mask_line_arr function works correctly.

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

    #flow0, flow1, and flow2 have shape (f, h, w)
    flow0 = tiff_arr[:, 0, :, :]
    flow1 = tiff_arr[:, 1, :, :]
    flow2 = tiff_arr[:, 2, :, :]

    threshold1 = 0.5
    threshold2 = 0.0
    threshold3 = 1.5
    threshold4 = 0.25

    def test_case_x(flowx: np.ndarray, thresholdx: float):
        """
        Tests whether the mask_line_arr function works correctly on a specific test case.

        Args:
            flowx (np.ndarray): Flow array of shape (frames, height, width).
            thresholdx (float): Threshold value to mask the array.

        Returns:
            None.
        """
        with patch("numpy.max") as mock_max, \
            patch("numpy.where") as mock_where:
            mock_max.side_effect = lambda x: x[0, 0, 0]
            mock_where.return_value = flowx

            result = kymograph.mask_line_arr(flowx, thresholdx)

            args_max, _ = mock_max.call_args
            assert np.array_equal(args_max[0], flowx)

            args_where, _ = mock_where.call_args
            assert np.array_equal(args_where[0], (flowx > thresholdx))
            assert np.array_equal(args_where[1], flowx[0, 0, 0])
            assert args_where[2] == 0

            mock_max.assert_called_once()
            mock_where.assert_called_once()

            assert result.shape == (f, h, w)
            assert result.shape == flowx.shape
            assert isinstance(result, np.ndarray)

    test_case_x(flow0, threshold1)
    test_case_x(flow0, threshold2)
    test_case_x(flow0, threshold3)
    test_case_x(flow0, threshold4)
    test_case_x(flow1, threshold1)
    test_case_x(flow1, threshold2)
    test_case_x(flow1, threshold3)
    test_case_x(flow1, threshold4)
    test_case_x(flow2, threshold1)
    test_case_x(flow2, threshold2)
    test_case_x(flow2, threshold3)
    test_case_x(flow2, threshold4)


def test_plot_basic_kymo(init_tiff: tuple, tmp_path):
    """
    Tests whether the plot_basic_kymo function works correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file.
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
        tmp_path (pathlib.Path): A path to a temporary directory (this is a fixture in Pytest).

    Returns:
        None.
    """
    img, info = init_tiff
    f, c, h, w = info
    tiff_arr = img.arr

    def dummy_optical_flow(channel: int):
        """
        A function that creates an array similar to (and with the same shape as) flow.optical_flow(tiff_arr, channel). This function is much
        faster than calling flow.optical_flow (which is why it's perfect for testing).

        Args:
            channel (int): The channel to process.

        Returns:
            A np.ndarray of shape (f-1, h, w, 2).
        """
        arr_channel = tiff_arr[:, channel, :, :]

        dummy_dx = arr_channel[1:] - arr_channel[:-1]
        dummy_dy = arr_channel[1:] - arr_channel[:-1]

        return np.stack([dummy_dx, dummy_dy], axis=-1)

    #flow0, flow1, and flow2 have shape (f-1, h, w, 2), which is the same shape as flow.optical_flow(tiff_arr, n) (where n is 0, 1, or 2)
    flow0 = dummy_optical_flow(0)
    flow1 = dummy_optical_flow(1)
    flow2 = dummy_optical_flow(2)

    threshold1 = 0.5
    threshold2 = 0.0
    threshold3 = 1.5
    threshold4 = 0.25

    WRITE SAVE_PATHS HERE!

    def test_case_x(flowx: np.ndarray, save_path_x: str, thresholdx: float):
        """
        Tests whether the plot_basic_kymo function works correctly on a specific test case.

        Args:
            flowx (np.ndarray): Flow array of shape (frames-1, height, width, 2).
            save_path_x (str): Path to save the plot. If None, the plot will be displayed instead of saved.
            thresholdx (float): Threshold value to mask the array.

        Returns:
            None.
        """