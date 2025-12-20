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


def test_create_color_wheel(init_tiff: tuple):
    """
    Tests whether the create_color_wheel function works correctly.

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

    size0 = 0
    size1 = 100
    size2 = 50
    size3 = 400
    size4 = 200

    def test_case_x(sizex: int):
        """
        Tests whether the create_color_wheel function works correctly on a specific test case.

        Args:
            sizex (int): Radius of the color wheel in pixels.

        Returns:
            None.
        """
        if sizex == 0:
            y, x = np.ogrid[-1 : 1 : 200 * 1j, -1 : 1 : 200 * 1j]
        else:
            y, x = np.ogrid[-1 : 1 : sizex * 1j, -1 : 1 : sizex * 1j]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        hue = (theta + np.pi) / (2 * np.pi)
        saturation = np.clip(r, 0, 1)
        value = np.ones_like(r)
        mask = r <= 1
        hsv = np.stack([hue, saturation, value], axis=-1)
        rgb = hsv_to_rgb(hsv)
        rgb[~mask] = 1

        expected_rgb = rgb
        expected_mask = mask

        if sizex == 0:
            result_rgb, result_mask = heatmap.create_color_wheel()
            result_rgb_200size, result_mask_200size = heatmap.create_color_wheel(200)
            assert np.array_equal(result_rgb, result_rgb_200size)
            assert np.array_equal(result_mask, result_mask_200size)

            assert result_rgb.shape == (200, 200, 3)
            assert result_mask.shape == (200, 200)

        else:
            result_rgb, result_mask = heatmap.create_color_wheel(sizex)

            assert result_rgb.shape == (sizex, sizex, 3)
            assert result_mask.shape == (sizex, sizex)

        assert result_rgb.shape == expected_rgb.shape
        assert np.array_equal(result_rgb, expected_rgb) 
        assert result_mask.shape == expected_mask.shape
        assert np.array_equal(result_mask, expected_mask) 

    test_case_x(size0)
    test_case_x(size1)
    test_case_x(size2)
    test_case_x(size3)
    test_case_x(size4)


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

    def dummy_polar_frame(channel: int, frame: int):
        """
        A function that creates an array of shape (h, w, 2).

        Args:
            channel (int): The channel to process.
            frame (int): The frame to process.

        Returns:
            A np.ndarray of shape (h, w, 2).
        """
        arr_channel = tiff_arr[:, channel, :, :]

        dummy_r = np.random.uniform(0, 0.1, size=(f-1, h, w))
        dummy_theta = np.random.uniform(0, 0.1, size=(f-1, h, w))

        all_frames = np.stack([dummy_r, dummy_theta], axis=-1)

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
        assert np.allclose(result, expected_result)

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
    fps3 = 18
    fps4 = 20

    def test_case_x(titlex: str, fpsx: int):
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
            # polar_to_heatmap_return has shape (f-1, h, w, 3), which is the same shape as polar_to_heatmap(polar_arr[0]), where polar_arr = arrx_0
            polar_to_heatmap_return = np.repeat(arrx_0[:, :, :1], 3, axis=-1)
            # create_color_wheel_rgb_return has shape (300, 300, 3), which is the same shape as rgb, where (rgb, mask) = create_color_wheel(300)
            create_color_wheel_color_wheel_return = np.random.rand(300, 300, 3)
            # create_color_wheel_mask_return has shape (300, 300), which is the same shape as mask, where (rgb, mask) = create_color_wheel(300)
            create_color_wheel_mask_return = np.random.rand(300, 300)

            mock_convert_stack_to_polar.return_value = arrx_0
            mock_figure.return_value = mock_fig
            mock_fig.add_gridspec.return_value = mock_gs
            mock_fig.add_subplot.side_effect = [mock_ax_main, mock_ax_wheel]
            mock_polar_to_heatmap.return_value = polar_to_heatmap_return
            mock_ax_main.imshow.return_value = mock_im
            mock_ax_main.set_title.return_value = mock_title_text
            mock_create_color_wheel.return_value = (create_color_wheel_color_wheel_return, create_color_wheel_mask_return)
            mock_funcanimation.return_value = mock_anim
            mock_ffmpegwriter.return_value = mock_writer

            heatmap.plot_heatmap(arrx, titlex, output_pathx, fpsx)

            num_frames = arrx_0.shape[0]

            args_fig_add_gridspec, kwargs_fig_add_gridspec = mock_fig.add_gridspec.call_args
            assert args_fig_add_gridspec[0] == 1
            assert args_fig_add_gridspec[1] == 2
            assert kwargs_fig_add_gridspec['width_ratios'] == [3, 1]
            assert kwargs_fig_add_gridspec['wspace'] == 0.3

            for i, call_argsx in enumerate(mock_fig.add_subplot.call_args_list):
                assert np.array_equal(call_argsx[0][0], mock_gs[i])

            # first polar_to_heatmap call
            first_args_polar_to_heatmap, _ = mock_polar_to_heatmap.call_args_list[0]
            assert first_args_polar_to_heatmap[0] == arrx_0[0]

            args_ax_main_imshow, kwargs_ax_main_imshow = mock_ax_main.imshow.call_args
            assert args_ax_main_imshow[0] == polar_to_heatmap_return
            assert kwargs_ax_main_imshow['origin'] == "lower"

            args_ax_wheel_imshow, kwargs_ax_wheel_imshow = mock_ax_wheel.imshow.call_args
            assert np.array_equal(args_ax_wheel_imshow[0], create_color_wheel_color_wheel_return)
            assert np.array_equal(kwargs_ax_wheel_imshow['extent'], [-1, 1, -1, 1])

            angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
            for i, angle_deg in enumerate(angles_deg):
                angle_rad = np.radians(angle_deg)
                x = 1.15 * np.cos(angle_rad)
                y = 1.15 * np.sin(angle_rad)
                args_ax_wheel_text, kwargs_ax_wheel_text = mock_ax_wheel.text.call_args[i]
                assert args_ax_wheel_text[0] == x
                assert args_ax_wheel_text[1] == y
                assert args_ax_wheel_text[2] == f"{angle_deg}°"
                assert kwargs_ax_wheel_text['ha'] == "center"
                assert kwargs_ax_wheel_text['va'] == "center"
                assert kwargs_ax_wheel_text['fontsize'] == 10

            print_message = f"Creating animation with {num_frames} frames..."

            args_funcanimation, kwargs_funcanimation = mock_funcanimation.call_args
            assert args_funcanimation[0] == mock_fig
            assert callable(args_funcanimation[1])
            assert kwargs_funcanimation['frames'] == num_frames
            assert kwargs_funcanimation['interval'] == 50
            assert kwargs_funcanimation['blit'] == True

            _, kwargs_ffmpegwriter = mock_ffmpegwriter.call_args
            assert kwargs_ffmpegwriter['fps'] == fpsx
            assert kwargs_ffmpegwriter['metadata'] == dict(artist="Matplotlib")
            assert kwargs_ffmpegwriter['bitrate'] == 1800

            args_anim_save, kwargs_anim_save = mock_anim.save.call_args
            assert args_anim_save[0] == output_pathx
            assert kwargs_anim_save['writer'] == mock_writer

            mock_convert_stack_to_polar.assert_called_once_with(arrx_0)
            mock_figure.assert_called_once_with(figsize=(14, 8))
            mock_fig.add_gridspec.assert_called_once()
            assert mock_fig.add_subplot.call_count == 2
            mock_ax_main.set_xlabel.assert_called_once_with("Width")
            mock_ax_main.set_ylabel.assert_called_once_with("Height")
            mock_polar_to_heatmap.assert_called_once()
            mock_ax_main.imshow.assert_called_once()
            mock_ax_main.set_title.assert_called_once_with(f"{titlex} - Frame 0/{num_frames-1}")
            mock_create_color_wheel.assert_called_once_with(300)
            mock_ax_wheel.imshow.assert_called_once()
            mock_ax_wheel.set_aspect.assert_called_once_with("equal")
            assert mock_ax_wheel.text.call_count == len(angles_deg)
            mock_ax_wheel.set_xlim.assert_called_once_with(-1.4, 1.4)
            mock_ax_wheel.set_ylim.assert_called_once_with(-1.4, 1.4)
            mock_ax_wheel.axis.assert_called_once_with("off")
            mock_tight_layout.assert_called_once_with()
            mock_print.assert_called_once_with(print_message)
            mock_funcanimation.assert_called_once()
            mock_ffmpegwriter.assert_called_once()
            mock_anim.save.assert_called_once()

    test_case_x(title1, fps1)
    test_case_x(title1, fps2)
    test_case_x(title1, fps3)
    test_case_x(title1, fps4)
    test_case_x(title2, fps1)
    test_case_x(title2, fps2)
    test_case_x(title2, fps3)
    test_case_x(title2, fps4)