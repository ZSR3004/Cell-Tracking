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
from matplotlib.colors import hsv_to_rgb
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

    # frame_stack0, frame_stack1, and frame_stack2 have shape (f-1, h, w, 2)
    frame_stack0 = dummy_optical_flow(0)
    frame_stack1 = dummy_optical_flow(1)
    frame_stack2 = dummy_optical_flow(2)

    def test_case_x(frame_stackx: np.ndarray):
        """
        Tests whether the convert_stack_to_polar function works correctly on a specific test case.

        Args:
            frame_stackx (np.ndarray): The numpy array representation
            of the TIFF file. Shape is (f-1, h, w, 2), where 2 is (dx, dy).

        Returns:
            None.
        """
        x = frame_stackx[..., 0]
        y = frame_stackx[..., 1]
        r = np.sqrt(x**2 + y**2)
        max_f = np.max(np.abs(frame_stackx))
        r_norm = r / (np.sqrt(2) * max_f)
        theta = np.arctan2(y, x)
        expected_result = np.stack([r_norm, theta], axis=-1)

        result = heatmap.convert_stack_to_polar(frame_stackx)

        assert result.shape == (f-1, h, w, 2)
        assert result.shape == frame_stackx.shape
        assert result.shape == expected_result.shape
        assert np.array_equal(result, expected_result)

    test_case_x(frame_stack0)
    test_case_x(frame_stack1)
    test_case_x(frame_stack2)


def test_create_color_wheel():
    """
    Cont when I get the answers to these questions:
    What is supposed to be in the returns part of the docstring?
    """


def test_polar_to_heatmap(init_tiff: tuple):
    """
    Tests whether the polar_to_heatmap function works correctly.

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

    def dummy_polar_frame(frame: int, channel: int):
        """
        A function that creates an array of shape (h, w, 2).

        Args:
            channel (int): The channel to process.

        Returns:
            A np.ndarray of shape (h, w, 2).
        """
        arr_channel = tiff_arr[:, channel, :, :]

        dummy_dx = arr_channel[1:] - arr_channel[:-1]
        dummy_dy = arr_channel[1:] - arr_channel[:-1]

        all_frames = np.stack([dummy_dx, dummy_dy], axis=-1)

        return all_frames[frame, ...]

    #these all have shape (h, w, 2)
    polar_frame_channel0_firstframe = dummy_polar_frame(0, 0)
    polar_frame_channel0_middleframe = dummy_polar_frame(0, f//2)
    polar_frame_channel0_lastframe = dummy_polar_frame(0, -1)
    polar_frame_channel1_firstframe = dummy_polar_frame(1, 0)
    polar_frame_channel1_middleframe = dummy_polar_frame(1, f//2)
    polar_frame_channel1_lastframe = dummy_polar_frame(1, -1)
    polar_frame_channel2_firstframe = dummy_polar_frame(2, 0)
    polar_frame_channel2_middleframe = dummy_polar_frame(2, f//2)
    polar_frame_channel2_lastframe = dummy_polar_frame(2, -1)

    def test_case_x(polar_frame_channelx_xframe: np.ndarray):
        """
        Tests whether the polar_to_heatmap function works correctly on a specific test case.

        Args:
            polar_frame_channelx_xframe (np.ndarray): Array of shape (height, width, 2), where last dim is (r, theta).

        Returns:
            None.
        """
        r = polar_frame_channelx_xframe[:, :, 0]
        theta = polar_frame_channelx_xframe[:, :, 1]
        hue = (theta + np.pi) / (2 * np.pi)
        saturation = r
        value = np.ones_like(r)
        hsv = np.stack([hue, saturation, value], axis=-1)
        rgb = hsv_to_rgb(hsv)

        expected_result = rgb

        result = heatmap.polar_to_heatmap(polar_frame_channelx_xframe)

        assert result.shape == (h, w, 3)
        assert result.shape == expected_result.shape
        assert np.array_equal(result, expected_result)

    test_case_x(polar_frame_channel0_firstframe)
    test_case_x(polar_frame_channel0_middleframe)
    test_case_x(polar_frame_channel0_lastframe)
    test_case_x(polar_frame_channel1_firstframe)
    test_case_x(polar_frame_channel1_middleframe)
    test_case_x(polar_frame_channel1_lastframe)
    test_case_x(polar_frame_channel2_firstframe)
    test_case_x(polar_frame_channel2_middleframe)
    test_case_x(polar_frame_channel2_lastframe)


def test_plot_heatmap(init_tiff: tuple, tmp_path):
    """
    Tests whether the plot_heatmap function works correctly.

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

    # Making flowx
    dummy_dx = tiff_arr[1:] - tiff_arr[:-1]
    dummy_dy = tiff_arr[1:] - tiff_arr[:-1] 

    # arrx has shape (f-1, c, h, w, 2), which is the shape of plot_heatmap's input array arr
    arrx = np.stack([dummy_dx, dummy_dy], axis=-1)

    title1 = "plot_heatmap0"
    title2 = "Test Title"

    output_pathx = tmp_path / "arrx"

    fps1 = 5
    fps2 = 50
    fps3 = 20

    def test_case_x(titlex: str, fpsx: int = 20):
        """
        Tests whether the plot_heatmap function works correctly on a specific test case.

        Args:
            titlex (str): Title for the heatmap.
            fpsx (int): Fps of the output heatmap video.

        Returns:
            None.
        """
        with (patch("src.cell_tracking.heatmap.convert_stack_to_polar") as mock_convert_stack_to_polar,
            patch("matplotlib.pyplot.figure") as mock_figure,
            patch("src.cell_tracking.heatmap.polar_to_heatmap") as mock_polar_to_heatmap,
            patch("src.cell_tracking.heatmap.create_color_wheel") as mock_create_color_wheel,
            patch("builtins.print") as mock_print,
            patch("src.cell_tracking.saving.animation.FuncAnimation") as mock_funcanimation,
            patch("src.cell_tracking.saving.animation.FFMpegWriter") as mock_ffmpegwriter,
            patch("matplotlib.pyplot.tight_layout") as mock_tight_layout
        ):
            mock_fig = MagicMock()
            mock_gs = MagicMock()
            mock_ax_main = MagicMock()
            mock_im = MagicMock()
            mock_title_text = MagicMock()
            mock_ax_wheel = MagicMock()
            mock_anim = MagicMock()
            mock_writer = MagicMock()

            # arrx_0 has shape (f-1, h, w, 2)
            arrx_0 = arrx[:, 0]

            mock_convert_stack_to_polar.return_value = arrx_0
            mock_figure.return_value = mock_fig
            mock_fig.add_gridspec.return_value = mock_gs
            mock_fig.add_subplot.side_effect = [mock_ax_main, mock_ax_wheel]
            # mock_polar_to_heatmap.return_value has shape (f-1, h, w, 3), which is the same shape as polar_to_heatmap(polar_arr[0]), where polar_arr = arrx_0
            mock_polar_to_heatmap.return_value = np.repeat(arrx_0[:, :, :1], 3, axis=-1)
            mock_ax_main.imshow.return_value = mock_im
            mock_ax_main.set_title.return_value = mock_title_text
            ADD WHEN I KNOW THE OUTPUTS OF CREATE COLOR WHEEL
            mock_funcanimation.return_value = mock_anim
            mock_ffmpegwriter.return_value = mock_writer

            num_frames = arrx_0.shape[0]

            args_fig_add_gridspec, kwargs_fig_add_gridspec = mock_fig.add_gridspec.call_args
            assert args_fig_add_gridspec[0] == 1
            assert args_fig_add_gridspec[1] == 2
            assert kwargs_fig_add_gridspec["width_ratios"] == [3, 1]
            assert kwargs_fig_add_gridspec["wspace"] == 0.3

            for i, call_argsx in enumerate(mock_fig.add_subplot.call_args_list):
                assert np.array_equal(call_argsx[0][0], mock_gs[i])

            mock_convert_stack_to_polar.assert_called_once_with(arrx_0)
            mock_figure.assert_called_once_with(figsize=(14, 8))

            """
            Assert args of:
            ax_main.set_xlabel
            ax_main.set_ylabel
            polar_to_heatmap (called multiple times)
            ax_main.imshow
            ax_main.set_title
            create_color_wheel
            ax_wheel.imshow
            ax_wheel.set_aspect
            np.radians (called multiple times)
            np.cos (called multiple times)
            np.sin (called multiple times)
            ax_wheel.text (called multiple times)
            ax_wheel.set_xlim
            ax_wheel.set_ylim
            ax_wheel.axis
            plt.tight_layout
            polar_to_heatmap (in update) (called multiple times)
            im.set_array (in update) (called multiple times)
            title_text.set_text (in update) (called multiple times)
            mock_print (printing)
            FuncAnimation
            FFMpegWriter
            anim.save
            """

            """
            Assert called:
            fig.add_gridspec
            fig.add_subplot (called multiple times)
            ax_main.set_xlabel
            ax_main.set_ylabel
            polar_to_heatmap (called multiple times)
            ax_main.imshow
            ax_main.set_title
            fig.add_subplot (called multiple times)
            create_color_wheel
            ax_wheel.imshow
            ax_wheel.set_aspect
            np.radians (called multiple times)
            np.cos (called multiple times)
            np.sin (called multiple times)
            ax_wheel.text (called multiple times)
            ax_wheel.set_xlim
            ax_wheel.set_ylim
            ax_wheel.axis
            plt.tight_layout
            polar_to_heatmap (in update) (called multiple times)
            im.set_array (in update) (called multiple times)
            title_text.set_text (in update) (called multiple times)
            mock_print (printing)
            FuncAnimation
            FFMpegWriter
            anim.save
            """



            """
            This is just a template. delete after.
            convert_stack_to_polar
            plt.figure
            fig.add_gridspec
            fig.add_subplot (called multiple times)
            ax_main.set_xlabel
            ax_main.set_ylabel
            polar_to_heatmap (called multiple times)
            ax_main.imshow
            ax_main.set_title
            fig.add_subplot (called multiple times)
            create_color_wheel
            ax_wheel.imshow
            ax_wheel.set_aspect
            np.radians (called multiple times)
            np.cos (called multiple times)
            np.sin (called multiple times)
            ax_wheel.text (called multiple times)
            ax_wheel.set_xlim
            ax_wheel.set_ylim
            ax_wheel.axis
            plt.tight_layout
            polar_to_heatmap (in update) (called multiple times)
            im.set_array (in update) (called multiple times)
            title_text.set_text (in update) (called multiple times)
            mock_print (printing)
            FuncAnimation
            FFMpegWriter
            anim.save
            """

#have some not have fpsx!

# note: when working with stuff of shape like (f-1, c, h, w, 2), use stuff i wrote for test_plot_basic_kymo
