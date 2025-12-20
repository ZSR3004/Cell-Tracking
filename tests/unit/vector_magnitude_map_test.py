import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest, tifffile
from unittest.mock import patch, Mock, MagicMock, ANY
from multiprocessing import Pool, cpu_count
from src.cell_tracking import tiffclass as tiff
import numpy as np

TIFF_PATHS = ["datasets/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif"]


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


def test_create_vector_field_video(init_tiff: tuple):
    """
    Tests whether the create_vector_field_video function works correctly.

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

    def dummy_arr(channel: int):
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

    # arr0, arr1, and arr2 have shape (f-1, h, w, 2)
    arr0 = dummy_arr(0)
    arr1 = dummy_arr(1)
    arr2 = dummy_arr(2)

    name1 = "create_vector_field_video_test"
    name2 = "Test Name"

    og_arr1 = tiff_arr
    og_arr2 = None

    step1 = 10
    step2 = 40
    step3 = 20

    scale1 = 250
    scale2 = 800
    scale3 = 500

    color1 = 'red'
    color2 = 'green'
    color3 = 'blue'

    fps1 = 6
    fps2 = 20
    fps3 = 10

    figsize1 = (6, 10)
    figsize2 = (14, 14)
    figsize3 = (12, 8)

    title1 = "Video Title"
    title2 = "test_title0"
    title3 = None

    flag1 = 'f'
    flag2 = 't'
    flag3 = ''

    def test_case_x(namex: str, arrx: np.ndarray, og_arrx, stepx: int, scalex: int, colorx: str, fpsx: int, figsizex: tuple, titlex, flagx: str):
        """
        Tests whether the create_vector_field_video function works correctly on a specific test case.

        Args:
            namex (str): Name of the video file to save.
            arrx (np.ndarray): Optical flow array of shape (f-1, h, w, 2) where f-1 is the number of frames,
                            h is height, w is width, and the last dimension contains the flow vectors (dx, dy).
            og_arrx (np.ndarray): Original image frames array of shape (f, c, h, w). Default is None.
            stepx (int): Step size for downsampling the flow vectors for visualization. Default is 20.
            scalex (int): Scale factor for the quiver arrows. Default is 500.
            colorx (str): Color of the arrows. Default is 'blue'.
            fpsx (int): Frames per second for the video. Default is 10.
            figsizex (tuple): Figure size in inches (width, height). Default is (12, 8).
            titlex (str): Title of the video. Default is None.
            flagx (str): Flag to determine if the video should be saved ('f' for flow, 't' for trajectory). Default is None.

        Returns:
            None
        """
        with (
            patch("matplotlib.pyplot.subplots") as mock_subplots,
            patch("src.cell_tracking.saving.save_vector_video") as mock_save_vector_video,
            patch("matplotlib.pyplot.close") as mock_close
        ):
            mock_fig = MagicMock()
            mock_ax = MagicMock()

            mock_subplots.return_value = (mock_fig, mock_ax)
            
            T, H, W, _ = arrx.shape
            Y, X = np.mgrid[0:H:stepx, 0:W:stepx]

            _ , kwargs_subplots = mock_subplots.call_args
            assert kwargs_subplots["figsize"] == (12, 6)




            mock_ax.set_xlim.assert_called_once_with(0, W)
            mock_ax.set_ylim.assert_called_once_with(H, 0)
            mock_ax.set_xlabel.assert_called_once_with("X")
            mock_ax.set_ylabel.assert_called_once_with("Y")
            mock_ax.set_title.assert_called_once_with("Optical Flow")
            mock_ax.set_aspect.assert_called_once_with('equal')
            mock_ax.axis.assert_called_once_with('off')

            """
            Args:
            ax.imshow
            ax.quiver
            saving.save_vector_video
            plt.close
            """

            """
            Assert called:
            plt.subplots
            ax.imshow
            ax.quiver
            saving.save_vector_video
            plt.close
            """