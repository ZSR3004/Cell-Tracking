import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest, tifffile
import numpy as np
from unittest.mock import patch, Mock, MagicMock, call
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.cell_tracking import tiffclass as tiff
from src.cell_tracking import kymograph as kymo

TIFF_PATHS = [
        "datasets/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif"
]


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
            #mock_linalg_norm_flow has shape (f-1, h, w), which is the same shape as np.linalg.norm(flowx, axis=-1)
            mock_linalg_norm_flow = flowx[..., 0]
            mock_linalg_norm.return_value = mock_linalg_norm_flow
            #mock_median.side_effect returns an array of shape (w), which is the same shape as np.median(mag_per_frame[i, :, :], axis=0) (note: the shape of mag_per_frame[i, :, :] is (h, w))
            mock_median.side_effect = lambda arr1, **axis1: arr1[0, :]

            result = kymo.flatten_arr(flowx)

            args_linalg_norm, kwargs_linalg_norm = mock_linalg_norm.call_args
            assert np.array_equal(args_linalg_norm[0], flowx)
            assert kwargs_linalg_norm["axis"] == -1

            for i, call_args_kwargs in enumerate(mock_median.call_args_list):
                args_median, kwargs_median = call_args_kwargs
                assert np.array_equal(args_median[0], mock_linalg_norm_flow[i, :, :])
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

    #all the flows below have shape (f-1, w) (which is the same shape as flatten_arr's output)
    flow0_first = tiff_arr[1:, 0, 0, :]
    flow0_middle = tiff_arr[1:, 0, h//2, :]
    flow0_last = tiff_arr[1:, 0, -1, :]
    flow1_first = tiff_arr[1:, 1, 0, :]
    flow1_middle = tiff_arr[1:, 1, h//2, :]
    flow1_last = tiff_arr[1:, 1, -1, :]
    flow2_first = tiff_arr[1:, 2, 0, :]
    flow2_middle = tiff_arr[1:, 2, h//2, :]
    flow2_last = tiff_arr[1:, 2, -1, :]

    threshold1 = 0.5
    threshold2 = 0.0
    threshold3 = 1.5
    threshold4 = 0.25

    def test_case_x(flowx: np.ndarray, thresholdx: float):
        """
        Tests whether the mask_line_arr function works correctly on a specific test case.

        Args:
            flowx (np.ndarray): Flow array of shape (frames-1, width).
            thresholdx (float): Threshold value to mask the array.

        Returns:
            None.
        """
        with patch("numpy.max") as mock_max, \
            patch("numpy.where") as mock_where:
            mock_max.side_effect = lambda x: x[0, 0]
            mock_where.return_value = flowx

            result = kymo.mask_line_arr(flowx, thresholdx)

            args_max, _ = mock_max.call_args
            assert np.array_equal(args_max[0], flowx)

            args_where, _ = mock_where.call_args
            assert np.array_equal(args_where[0], (flowx > thresholdx))
            assert np.array_equal(args_where[1], flowx[0, 0])
            assert args_where[2] == 0

            mock_max.assert_called_once()
            mock_where.assert_called_once()

            assert result.shape == (f-1, w)
            assert result.shape == flowx.shape
            assert isinstance(result, np.ndarray)

    test_case_x(flow0_first, threshold1)
    test_case_x(flow0_first, threshold2)
    test_case_x(flow0_first, threshold3)
    test_case_x(flow0_first, threshold4)
    test_case_x(flow0_middle, threshold1)
    test_case_x(flow0_middle, threshold2)
    test_case_x(flow0_middle, threshold3)
    test_case_x(flow0_middle, threshold4)
    test_case_x(flow0_last, threshold1)
    test_case_x(flow0_last, threshold2)
    test_case_x(flow0_last, threshold3)
    test_case_x(flow0_last, threshold4)
    test_case_x(flow1_first, threshold1)
    test_case_x(flow1_first, threshold2)
    test_case_x(flow1_first, threshold3)
    test_case_x(flow1_first, threshold4)
    test_case_x(flow1_middle, threshold1)
    test_case_x(flow1_middle, threshold2)
    test_case_x(flow1_middle, threshold3)
    test_case_x(flow1_middle, threshold4)
    test_case_x(flow1_last, threshold1)
    test_case_x(flow1_last, threshold2)
    test_case_x(flow1_last, threshold3)
    test_case_x(flow1_last, threshold4)
    test_case_x(flow2_first, threshold1)
    test_case_x(flow2_first, threshold2)
    test_case_x(flow2_first, threshold3)
    test_case_x(flow2_first, threshold4)
    test_case_x(flow2_middle, threshold1)
    test_case_x(flow2_middle, threshold2)
    test_case_x(flow2_middle, threshold3)
    test_case_x(flow2_middle, threshold4)
    test_case_x(flow2_last, threshold1)
    test_case_x(flow2_last, threshold2)
    test_case_x(flow2_last, threshold3)
    test_case_x(flow2_last, threshold4)


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

    #Making flowx
    dummy_dx = tiff_arr[1:] - tiff_arr[:-1]
    dummy_dy = tiff_arr[1:] - tiff_arr[:-1]

    #flowx has shape (f-1, c, h, w, 2), which is the shape of plot_basic_kymo's input array
    flowx = np.stack([dummy_dx, dummy_dy], axis=-1)

    threshold1 = 0.5
    threshold2 = 0.0
    threshold3 = 1.5
    threshold4 = 0.25

    flowx_threshold1_path = tmp_path / "flowx_threshold1_path.png"
    flowx_threshold2_path = tmp_path / "flowx_threshold2_path.png"
    flowx_threshold3_path = tmp_path / "flowx_threshold3_path.png"
    flowx_threshold4_path = tmp_path / "flowx_threshold4_path.png"

    def test_case_x(flowx: np.ndarray, thresholdx: float, save_path_x: str=None):
        """
        Tests whether the plot_basic_kymo function works correctly on a specific test case.

        Args:
            flowx (np.ndarray): Flow array of shape (frames-1, channel, height, width, 2).
            thresholdx (float): Threshold value to mask the array.
            save_path_x (str): Path to save the plot. If None, the plot will be displayed instead of saved.

        Returns:
            None.
        """
        with patch("src.cell_tracking.kymograph.flatten_arr") as mock_flatten_arr, \
            patch("src.cell_tracking.kymograph.mask_line_arr") as mock_mask_line_arr, \
            patch("matplotlib.pyplot.figure") as mock_figure, \
            patch("matplotlib.pyplot.imshow") as mock_imshow, \
            patch("matplotlib.pyplot.title") as mock_title, \
            patch("matplotlib.pyplot.xlabel") as mock_xlabel, \
            patch("matplotlib.pyplot.ylabel") as mock_ylabel, \
            patch("matplotlib.pyplot.savefig") as mock_savefig, \
            patch("matplotlib.pyplot.close") as mock_close, \
            patch("matplotlib.pyplot.show") as mock_show:
            #mock_flatten_arr_flow has shape (f-1, w), which is the same shape as kymograph.flatten_arr(flowx)
            mock_flatten_arr_flow = flowx[:, 0, :, 0]
            mock_flatten_arr.return_value = mock_flatten_arr_flow
            #mock_mask_line_arr_flow returns an array of shape (f, h, w), which is the same shape as kymograph.flatten_arr(arr1) (where arr1 is an array of shape (f, h, w))
            mock_mask_line_arr.side_effect = lambda x: x

            def mask_boundary(channel_arr: np.ndarray, threshold: float=0.5):
                """
                Function that's inside plot_basic_kymo. This function is needed to create masked_line_arr1 and masked_line_arr2.
                """
                return kymo.mask_line_arr(kymo.flatten_arr(channel_arr))

            masked_line_arr1 = mask_boundary(flowx[:, 1, ...], threshold=thresholdx)
            masked_line_arr2 = mask_boundary(flowx[:, 2, ...], threshold=thresholdx)

            assert masked_line_arr1.shape == (f, h, w, 2)
            assert masked_line_arr2.shape == (f, h, w, 2)

            combined_data = np.zeros_like(masked_line_arr1)
            combined_data[masked_line_arr1 != 0] += 1
            combined_data[masked_line_arr2 != 0] += 2

            custom_green = (119/255, 237/255, 130/255)
            custom_magenta = (201/255, 107/255, 232/255)
            custom_blue = (18/255, 105/255, 204/255)
            colors = ['black', custom_green, custom_magenta, custom_blue]
            cmap = mcolors.ListedColormap(colors)

            kymo.plot_basic_kymo(flowx, save_path_x, thresholdx)

            flowx_1 = flowx[:, 1, ...]
            flowx_2 = flowx[:, 2, ...]

            for i, flowx_x in enumerate([flowx_1, flowx_2]):
                assert np.array_equal(mock_flatten_arr.call_args_list[i][0][0], flowx_x)

            flowx_1_flatten = kymo.flatten_arr(flowx_1)
            flowx_2_flatten = kymo.flatten_arr(flowx_2)

            for i, flowx_x_flatten in enumerate([flowx_1_flatten, flowx_2_flatten]):
                assert np.array_equal(mock_mask_line_arr.call_args_list[i][0][0], flowx_x_flatten)
                assert mock_mask_line_arr.call_args_list[i][1].get("threshold") == None
            
            _, kwargs_figure = mock_figure.call_args
            assert kwargs_figure["figsize"] == (10, 5)

            args_imshow, kwargs_imshow = mock_imshow.call_args
            assert np.array_equal(args_imshow[0], combined_data)
            assert kwargs_imshow["aspect"] == "auto"
            assert kwargs_imshow["cmap"] == cmap
            assert kwargs_imshow["vmin"] == 0
            assert kwargs_imshow["vmax"] == 3

            #Called: 2 times when kymo.plot_basic_kymo was called, 2 times when mask_boundary (in test_case_x) was called, and 2 times when kymo.flatten_arr was called to create flowx_1_flatten and flowx_2_flatten
            assert mock_flatten_arr.call_count == 6
            #Called: 2 times when kymo.plot_basic_kymo was called, and 2 times when mask_boundary (in test_case_x) was called
            assert mock_mask_line_arr.call_count == 4
            mock_figure.assert_called_once()
            mock_imshow.assert_called_once()
            mock_title.assert_called_once_with("Overlay: Left (Green), Right (Magenta), Overlap (Blue)")
            mock_xlabel.assert_called_once_with('Position')
            mock_ylabel.assert_called_once_with('Time')

            if save_path_x:
                args_savefig, kwargs_savefig = mock_savefig.call_args
                assert args_savefig[0] == save_path_x
                assert kwargs_savefig["bbox_inches"] == "tight"
                assert kwargs_savefig["dpi"] == 300

                mock_savefig.assert_called_once()
                mock_close.assert_called_once_with()
                mock_show.assert_not_called()
            else:
                mock_show.assert_called_once_with()
                mock_savefig.assert_not_called()
                mock_close.assert_not_called()

    test_case_x(flowx, threshold1, flowx_threshold1_path)   #flowx, threshold1 save
    test_case_x(flowx, threshold1, None)                    #flowx, threshold1 show
    test_case_x(flowx, threshold2, flowx_threshold2_path)   #flowx, threshold2 save
    test_case_x(flowx, threshold2, None)                    #flowx, threshold2 show
    test_case_x(flowx, threshold3, flowx_threshold3_path)   #flowx, threshold3 save
    test_case_x(flowx, threshold3, None)                    #flowx, threshold3 show
    test_case_x(flowx, threshold4, flowx_threshold4_path)   #flowx, threshold4 save
    test_case_x(flowx, threshold4, None)                    #flowx, threshold4 show