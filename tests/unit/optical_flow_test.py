import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2, gc, pytest, tifffile
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

    combine_flow_0 = flow.combine_flows([flow0, flow1])
    combine_flow_1 = flow.combine_flows([flow1, flow2])
    combine_flow_2 = flow.combine_flows([flow0, flow2])

    assert combine_flow_0.shape == (f - 1, 3, h, w, 2)
    assert combine_flow_1.shape == (f - 1, 3, h, w, 2)
    assert combine_flow_2.shape == (f - 1, 3, h, w, 2)

    assert isinstance(combine_flow_0, np.ndarray)
    assert isinstance(combine_flow_1, np.ndarray)
    assert isinstance(combine_flow_2, np.ndarray)


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
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }
    kwargs2 = {
        "pyr_scale": 0.75,
        "levels": 5,
        "winsize": 17,
        "iterations": 5,
        "poly_n": 10,
        "poly_sigma": 1.4,
        "flags": 1,
    }
    kwargs3 = {"levels": 5, "winsize": 17, "poly_n": 10, "flags": 1}
    kwargs4 = {}

    def test_case_x(channelx: int, **kwargsx):
        """
        Tests whether the optical_flow function works correctly on a specific test case.

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

            mock_pool_instance = mock_pool.return_value.__enter__.return_value
            mock_pool_instance.map.side_effect = lambda func, arr1: [np.zeros((x[0].shape[0], x[0].shape[1], 2)) for x in arr1]

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
    test_case_x(0, **kwargs4)
    test_case_x(1, **kwargs4)
    test_case_x(2, **kwargs4)


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

    kwargs1 = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }
    kwargs2 = {
        "pyr_scale": 0.75,
        "levels": 5,
        "winsize": 17,
        "iterations": 5,
        "poly_n": 10,
        "poly_sigma": 1.4,
        "flags": 1,
    }
    kwargs3 = {"levels": 5, "winsize": 17, "poly_n": 10, "flags": 1}
    kwargs4 = {}

    def test_case_x(**kwargsx):
        """
        Tests whether the calculate_optical_flow function works correctly.

        Args:
            **kwargsx: Additional keyword arguments passed to `optical_flow` for Farneback parameters:
                - pyr_scale (float)
                - levels (int)
                - winsize (int)
                - iterations (int)
                - poly_n (int)
                - poly_sigma (float)
                - flags (int)

        Returns:
            None.
        """
        with patch("src.cell_tracking.optical_flow.optical_flow") as mock_optflow, \
            patch("src.cell_tracking.optical_flow.combine_flows") as mock_combine:
            mock_optflow.return_value = np.zeros((f-1, h, w, 2))
            mock_combine.return_value = np.zeros((f-1, 3, h, w, 2))

            result = flow.calculate_optical_flow(tiff_arr, **kwargsx)

            first_optflow_args, first_optflow_kwargs = mock_optflow.call_args_list[0]
            assert np.array_equal(tiff_arr, first_optflow_args[0])
            assert first_optflow_args[1] == 1
            assert first_optflow_kwargs == kwargsx

            second_optflow_args, second_optflow_kwargs = mock_optflow.call_args_list[1]
            assert np.array_equal(tiff_arr, second_optflow_args[0])
            assert second_optflow_args[1] == 2
            assert second_optflow_kwargs == kwargsx

            combine_args, _ = mock_combine.call_args
            assert np.array_equal(combine_args[0], [np.zeros((f-1, h, w, 2)), np.zeros((f-1, h, w, 2))])

            assert np.array_equal(result, np.zeros((f-1, 3, h, w, 2)))
            assert result.shape == (f-1, 3, h, w, 2)
            assert isinstance(result, np.ndarray)

            assert mock_optflow.call_count == 2
            mock_combine.assert_called_once()

    test_case_x(**kwargs1)
    test_case_x(**kwargs2)
    test_case_x(**kwargs3)
    test_case_x(**kwargs4)


def test_show_flow(init_tiff, tmp_path):
    """
    Tests the show_flow function.

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

    dummy_optical_flow_channel0 = dummy_optical_flow(0)
    dummy_optical_flow_channel1 = dummy_optical_flow(1)
    dummy_optical_flow_channel2 = dummy_optical_flow(2)

    first_frame_channel0 = dummy_optical_flow_channel0[0]
    middle_frame_channel0 = dummy_optical_flow_channel0[f//2]
    last_frame_channel0 = dummy_optical_flow_channel0[-1]
    first_frame_channel1 = dummy_optical_flow_channel1[0]
    middle_frame_channel1 = dummy_optical_flow_channel1[f//2]
    last_frame_channel1 = dummy_optical_flow_channel1[-1]
    first_frame_channel2 = dummy_optical_flow_channel2[0]
    middle_frame_channel2 = dummy_optical_flow_channel2[f//2]
    last_frame_channel2 = dummy_optical_flow_channel2[-1]

    first_frame_channel0_save_path = tmp_path / "first_frame_channel0.png"
    middle_frame_channel0_save_path = tmp_path / "middle_frame_channel0.png"
    last_frame_channel0_save_path = tmp_path / "last_frame_channel0.png"
    first_frame_channel1_save_path = tmp_path / "first_frame_channel1.png"
    middle_frame_channel1_save_path = tmp_path / "middle_frame_channel1.png"
    last_frame_channel1_save_path = tmp_path / "last_frame_channel1.png"
    first_frame_channel2_save_path = tmp_path / "first_frame_channel2.png"
    middle_frame_channel2_save_path = tmp_path / "middle_frame_channel2.png"
    last_frame_channel2_save_path = tmp_path / "last_frame_channel2.png"

    def test_case_x(flowx: np.ndarray,
                    titlex: str = "Optical Flow",
                    stepx: int = 25,
                    figsizex: int | int = (12, 6),
                    scalex: int = 200,
                    pivotx: str = "tail",
                    colorx: str = "blue",
                    x_save_path: str = None):
        """
        Tests the show_flow function on a specific test case.

        Args:
            flow (np.ndarray): Optical flow array of shape (H, W, 2) where H is height, W is width,
                               and the last dimension contains the flow vectors (dx, dy).
            title (str): Title of the plot. Default is 'Optical Flow'.
            step (int): Step size for downsampling the flow vectors for visualization. Default is 25.
            figsize (tuple): Size of the figure in inches (width, height). Default is (12, 6).
            scale (float): Scale factor for the quiver arrows. Default is 200.
            pivot (str): Pivot point for the arrows. Default is 'tail'.
            color (str): Color of the arrows. Default is 'white'.
            save_path (str, optional): If provided, saves the image to this path.

        Returns:
            None.
        """
        Y, X = np.mgrid[0 : flowx.shape[0] : stepx, 0 : flowx.shape[1] : stepx]
        U = flowx[::stepx, ::stepx, 0]  # dx
        V = flowx[::stepx, ::stepx, 1]  # dy

        with patch("matplotlib.pyplot.figure") as mock_figure, \
            patch("matplotlib.pyplot.quiver") as mock_quiver, \
            patch("matplotlib.pyplot.title") as mock_title, \
            patch("matplotlib.pyplot.xlim") as mock_xlim, \
            patch("matplotlib.pyplot.ylim") as mock_ylim, \
            patch("matplotlib.pyplot.xlabel") as mock_xlabel, \
            patch("matplotlib.pyplot.ylabel") as mock_ylabel, \
            patch("matplotlib.pyplot.tight_layout") as mock_tight_layout, \
            patch("matplotlib.pyplot.savefig") as mock_savefig, \
            patch("matplotlib.pyplot.show") as mock_show:
            flow.show_flow(flowx, titlex, stepx, figsizex, scalex, pivotx, colorx, x_save_path)
            
            _, figure_kwargs = mock_figure.call_args
            assert figure_kwargs["figsize"] == figsizex

            quiver_args, quiver_kwargs = mock_quiver.call_args
            assert np.array_equal(quiver_args[0], X)
            assert np.array_equal(quiver_args[1], Y)
            assert np.array_equal(quiver_args[2], U)
            assert np.array_equal(quiver_args[3], V)
            assert quiver_kwargs["scale"] == scalex
            assert quiver_kwargs["pivot"] == pivotx
            assert quiver_kwargs["color"] == colorx

            title_args, _ = mock_title.call_args
            assert title_args[0] == titlex

            xlim_args, _ = mock_xlim.call_args
            assert xlim_args[0] == 0
            assert xlim_args[1] == flowx.shape[1]

            ylim_args, _ = mock_ylim.call_args
            assert ylim_args[0] == flowx.shape[0]
            assert ylim_args[1] == 0

            xlabel_args, _ = mock_xlabel.call_args
            assert xlabel_args[0] == "X"

            ylabel_args, _ = mock_ylabel.call_args
            assert ylabel_args[0] == "Y"

            if x_save_path:
                args_savefig, kwargs_savefig = mock_savefig.call_args
                assert args_savefig[0] == x_save_path
                assert kwargs_savefig["bbox_inches"] == "tight"

                mock_savefig.assert_called_once()
                mock_show.assert_not_called()
            else:
                mock_show.assert_called_once_with()
                mock_savefig.assert_not_called()

            mock_figure.assert_called_once()
            mock_quiver.assert_called_once()
            mock_title.assert_called_once()
            mock_xlim.assert_called_once()
            mock_ylim.assert_called_once()
            mock_xlabel.assert_called_once()
            mock_ylabel.assert_called_once()
            mock_tight_layout.assert_called_once_with()

            gc.collect()

    test_case_x(first_frame_channel0, "first_frame_channel0", 30, (14, 8), 300, "middle", "red", first_frame_channel0_save_path) #first_frame_channel0 save
    test_case_x(first_frame_channel0, stepx=15, figsizex=(10,10), scalex=150, pivotx="tail", colorx="blue", x_save_path=None) #first_frame_channel0 show
    test_case_x(middle_frame_channel0, titlex="Sample_Title", x_save_path=middle_frame_channel0_save_path) #middle_frame_channel0 save
    test_case_x(flowx=middle_frame_channel0, titlex="middle_frame_channel0", stepx=18, figsizex=(12,10), scalex=225, pivotx="tip", colorx="green", x_save_path=None) #middle_frame_channel0 show
    test_case_x(last_frame_channel0, titlex="last_frame_channel0", stepx=15, scalex=100, colorx="red", x_save_path=last_frame_channel0_save_path) #last_frame_channel0 save
    test_case_x(last_frame_channel0) #last_frame_channel0 show
    test_case_x(first_frame_channel1, x_save_path=first_frame_channel1_save_path) #first_frame_channel1 save
    test_case_x(first_frame_channel1, titlex="Optical Flow", stepx=25, figsizex=(12, 6), scalex=200, pivotx="tail", colorx="blue", x_save_path=None) #first_frame_channel1 show
    test_case_x(middle_frame_channel1, x_save_path=middle_frame_channel1_save_path) #middle_frame_channel1 save
    test_case_x(middle_frame_channel1, titlex="middle_frame_channel1") #middle_frame_channel1 show
    test_case_x(flowx=last_frame_channel1, titlex="last_frame_channel1", stepx=20, figsizex=(8,8), scalex=350, pivotx="middle", colorx="yellow", x_save_path=last_frame_channel1_save_path) #last_frame_channel1 save
    test_case_x(last_frame_channel1, scalex=240) #last_frame_channel1 show
    test_case_x(first_frame_channel2, x_save_path=first_frame_channel2_save_path) #first_frame_channel2 save
    test_case_x(first_frame_channel2, stepx=180, scalex=400, pivotx="middle", colorx="black") #first_frame_channel2 show
    test_case_x(middle_frame_channel2, titlex="middle_frame_channel2", figsizex=(6,8), scalex=200, pivotx="tip", colorx="orange", x_save_path=middle_frame_channel2_save_path) #middle_frame_channel2 save
    test_case_x(middle_frame_channel2, "middle_frame_channel2", 27, (10, 6), 250, "tip", "purple", None) #middle_frame_channel2 show
    test_case_x(last_frame_channel2, titlex="last_frame_channel2", figsizex=(12,6), x_save_path=last_frame_channel2_save_path) #last_frame_channel2 save
    test_case_x(last_frame_channel2, titlex="last_frame_channel2", stepx=30, figsizex=(10,14), scalex=180, pivotx="middle", colorx="black") #last_frame_channel2 show