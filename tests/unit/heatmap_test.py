import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2, pytest
from src.cell_tracking import heatmap
from src.cell_tracking import tiffclass as tiff
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from unittest.mock import patch, Mock, MagicMock

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


def test_vector_magnitude_heatmaps(init_tiff: tuple):
    """
    Tests whether the vector_magnitude_heatmaps function works correctly.

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

    def test_case_x(flowx: np.ndarray, normalizex: bool):
        """
        Tests whether the vector_magnitude_heatmaps function works correctly on a specific test case.

        Args:
            flowx (np.ndarray): Flow array of shape (frames-1, height, width, 2).
            normalizex (bool): If True, normalizes magnitudes to 0-255 range for visualization.

        Returns:
            None.
        """
        with patch("numpy.linalg.norm") as mock_linalg_norm, \
            patch("cv2.normalize") as mock_cv2_normalize:
            #fake_flow has shape (f, h, w), which is the same shape as np.linalg.norm(flowx, axis=-1)
            fake_flow = flowx[..., 0]
            mock_linalg_norm.return_value = fake_flow

            mock_cv2_normalize.side_effect = lambda frame1, *args, **kwargs: frame1

            result = heatmap.vector_magnitude_heatmaps(flowx, normalize=normalizex)

            if normalizex:
                i = 0
                for frame in fake_flow:
                    assert np.array_equal(mock_cv2_normalize.call_args_list[i][0][0], frame)
                    assert mock_cv2_normalize.call_args_list[i][0][1] == None
                    assert mock_cv2_normalize.call_args_list[i][0][2] == 0
                    assert mock_cv2_normalize.call_args_list[i][0][3] == 255
                    assert mock_cv2_normalize.call_args_list[i][0][4] == cv2.NORM_MINMAX
                    i += 1

                assert mock_cv2_normalize.call_count == fake_flow.shape[0]

            else:
                mock_cv2_normalize.assert_not_called()

            mock_linalg_norm_args, mock_linalg_norm_kwargs = mock_linalg_norm.call_args
            assert np.array_equal(mock_linalg_norm_args[0], flowx)
            assert mock_linalg_norm_kwargs["axis"] == -1

            mock_linalg_norm.assert_called_once()
            assert result.shape == (flowx.shape[0], flowx.shape[1], flowx.shape[2])
            assert result.shape == (f-1, h, w)
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.uint8

    test_case_x(flow0, True)
    test_case_x(flow0, False)
    test_case_x(flow1, True)
    test_case_x(flow1, False)
    test_case_x(flow2, True)
    test_case_x(flow2, False)


def test_save_heatmap_video(init_tiff: tuple, tmp_path):
    """
    Tests whether the save_heatmap_video function works correctly.

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

    fps1 = 10
    fps2 = 25
    fps3 = 100

    flow0_fps1_true_path = tmp_path / "flow0_fps1_true.mp4"
    flow0_fps1_false_path = tmp_path / "flow0_fps1_false.mp4"
    flow0_fps2_true_path = tmp_path / "flow0_fps2_true.mp4"
    flow0_fps2_false_path = tmp_path / "flow0_fps2_false.mp4"
    flow0_fps3_true_path = tmp_path / "flow0_fps3_true.mp4"
    flow0_fps3_false_path = tmp_path / "flow0_fps3_false.mp4"
    flow1_fps1_true_path = tmp_path / "flow1_fps1_true.mp4"
    flow1_fps1_false_path = tmp_path / "flow1_fps1_false.mp4"
    flow1_fps2_true_path = tmp_path / "flow1_fps2_true.mp4"
    flow1_fps2_false_path = tmp_path / "flow1_fps2_false.mp4"
    flow1_fps3_true_path = tmp_path / "flow1_fps3_true.mp4"
    flow1_fps3_false_path = tmp_path / "flow1_fps3_false.mp4"
    flow2_fps1_true_path = tmp_path / "flow2_fps1_true.mp4"
    flow2_fps1_false_path = tmp_path / "flow2_fps1_false.mp4"
    flow2_fps2_true_path = tmp_path / "flow2_fps2_true.mp4"
    flow2_fps2_false_path = tmp_path / "flow2_fps2_false.mp4"
    flow2_fps3_true_path = tmp_path / "flow2_fps3_true.mp4"
    flow2_fps3_false_path = tmp_path / "flow2_fps3_false.mp4"

    def test_case_x(flowx: np.ndarray, output_pathx: str, fpsx: int, normalizex: bool):
        """
        Tests whether the save_heatmap_video function works correctly on a specific test case.

        Args:
            flowx (np.ndarray): Flow array of shape (frames-1, height, width, 2).
            output_pathx (str): Path to save the MP4 video to.
            fpsx (int): Frames per second of the output video.
            normalizex (bool): Whether to normalize magnitudes per frame.

        Returns:
            None.
        """
        with patch("src.cell_tracking.heatmap.vector_magnitude_heatmaps") as mock_vector_mag_heatmaps, \
            patch("src.cell_tracking.heatmap.plt.subplots") as mock_plt_subplots, \
            patch("src.cell_tracking.heatmap.plt.close") as mock_plt_close, \
            patch("src.cell_tracking.heatmap.animation.FuncAnimation") as mock_funcanimation:
            mock_ani = MagicMock()
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_im = MagicMock()

            #This turns flowx from shape (f-1, h, w, 2) into shape (f-1, h, w)
            flowx_no_dx_dy = flowx[..., 0] 

            mock_vector_mag_heatmaps.return_value = flowx_no_dx_dy
            mock_funcanimation.return_value = mock_ani
            mock_plt_subplots.return_value = (mock_fig, mock_ax)
            mock_ax.imshow.return_value = mock_im

            heatmap.save_heatmap_video(flowx, output_pathx, fps=fpsx, normalize=normalizex)

            args_vector_mag_heatmap, kwargs_vector_mag_heatmap = mock_vector_mag_heatmaps.call_args
            assert np.array_equal(args_vector_mag_heatmap[0], flowx)
            assert kwargs_vector_mag_heatmap["normalize"] == normalizex

            args_ax_imshow, kwargs_ax_imshow = mock_ax.imshow.call_args
            assert np.array_equal(args_ax_imshow[0], flowx_no_dx_dy[0])
            assert kwargs_ax_imshow["cmap"] == 'jet'
            assert kwargs_ax_imshow["animated"] == True
            
            args_FuncAnimation, kwargs_FuncAnimation = mock_funcanimation.call_args
            assert args_FuncAnimation[0] == mock_fig
            assert callable(args_FuncAnimation[1])
            assert kwargs_FuncAnimation["frames"] == len(flowx_no_dx_dy)
            assert kwargs_FuncAnimation["interval"] == 1000 / fpsx
            assert kwargs_FuncAnimation["blit"] == True
        
            #ASSERTING CALLED ONCE
            mock_vector_mag_heatmaps.assert_called_once()
            mock_plt_subplots.assert_called_once_with()
            mock_ax.imshow.assert_called_once()
            mock_ax.axis.assert_called_once_with('off')
            mock_funcanimation.assert_called_once()
            mock_ani.save.assert_called_once_with(output_pathx, fps=fpsx, extra_args=['-vcodec', 'libx264'])
            mock_plt_close.assert_called_once_with(mock_fig)

    test_case_x(flow0, flow0_fps1_true_path, fps1, True)
    test_case_x(flow0, flow0_fps1_false_path, fps1, False)
    test_case_x(flow0, flow0_fps2_true_path, fps2, True)
    test_case_x(flow0, flow0_fps2_false_path, fps2, False)
    test_case_x(flow0, flow0_fps3_true_path, fps3, False)
    test_case_x(flow0, flow0_fps3_false_path, fps3, False)
    test_case_x(flow1, flow1_fps1_true_path, fps1, True)
    test_case_x(flow1, flow1_fps1_false_path, fps1, False)
    test_case_x(flow1, flow1_fps2_true_path, fps2, True)
    test_case_x(flow1, flow1_fps2_false_path, fps2, False)
    test_case_x(flow1, flow1_fps3_true_path, fps3, True)
    test_case_x(flow1, flow1_fps3_false_path, fps3, False)
    test_case_x(flow2, flow2_fps1_true_path, fps1, True)
    test_case_x(flow2, flow2_fps1_false_path, fps1, False)
    test_case_x(flow2, flow2_fps2_true_path, fps2, True)
    test_case_x(flow2, flow2_fps2_false_path, fps2, False)
    test_case_x(flow2, flow2_fps3_true_path, fps3, True)
    test_case_x(flow2, flow2_fps3_false_path, fps3, False)