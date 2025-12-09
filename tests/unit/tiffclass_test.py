import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2, pytest, gc
from unittest.mock import patch, Mock, MagicMock, ANY
from multiprocessing import Pool, cpu_count
from src.cell_tracking import tiffclass as tiff
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

TIFF_PATHS = [
        "datasets/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif"
]


@pytest.fixture(params=TIFF_PATHS)
def init_tiff(request: pytest.FixtureRequest) -> tiff.Tiff:
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
    img = tiff.imread(path)
    info = (img.shape[0], img.shape[1], img.shape[2], img.shape[3])
    return (tiff.Tiff(path), info)


def test_init(init_tiff: tuple):
    """
    Tests whether the Tiff class initializes correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file.
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Return:
        None
    """
    img, info = init_tiff
    f, c, h, w = info

    assert isinstance(img.arr, np.ndarray)
    assert hasattr(img, "path")
    assert hasattr(img, "timestamp")
    assert hasattr(img, "arr")

    assert img.arr.shape == (f, c, h, w)
    assert img.arr.shape[0] == f  # number of frames
    assert img.arr.shape[1] == c  # number of channels
    assert img.arr.shape[2] == h  # height
    assert img.arr.shape[3] == w  # width


def test_isolate_channel(init_tiff: tuple):
    """
    Tests whether the isolate_channel method works correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file.
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Return:
        None
    """
    img, info = init_tiff
    f, c, h, w = info

    channel_0 = img.isolate_channel(0)
    channel_1 = img.isolate_channel(1)
    channel_2 = img.isolate_channel(2)

    assert isinstance(channel_0, np.ndarray)
    assert isinstance(channel_1, np.ndarray)
    assert isinstance(channel_2, np.ndarray)

    assert channel_0.shape == (f, h, w)
    assert channel_1.shape == (f, h, w)
    assert channel_2.shape == (f, h, w)

    assert np.array_equal(channel_0, img.arr[:, 0, :, :])
    assert np.array_equal(channel_1, img.arr[:, 1, :, :])
    assert np.array_equal(channel_2, img.arr[:, 2, :, :])

    assert not np.array_equal(channel_0, channel_1)
    assert not np.array_equal(channel_1, channel_2)
    assert not np.array_equal(channel_0, channel_2)


def test_show_image(init_tiff: tuple, tmp_path):
    """
    Tests whether the show_image method works correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file:
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
        tmp_path (pathlib.Path): A path to a temporary directory (this is a fixture in Pytest).

    Return:
        None
    """
    img, info = init_tiff
    f, c, h, w = info

    image1 = img.arr[0, 2, :, :]
    image2 = img.arr[(f - 1) // 2, 0, :, :]
    image3 = img.arr[f - 1, 1, :, :]
    image4 = img.arr[0, 1, :, :]
    image5 = img.arr[(f - 1) // 2, 2, :, :]
    image6 = img.arr[f - 1, 0, :, :]

    image1_save_path = tmp_path / "image1_save.png"
    image2_save_path = tmp_path / "image2_save.png"
    image3_save_path = tmp_path / "image3_save.png"
    image4_save_path = tmp_path / "image4_save.png"
    image5_save_path = tmp_path / "image5_save.png"
    image6_save_path = tmp_path / "image6_save.png"

    def test_case_x(imagex: np.ndarray, titlex="Image", figsizex=(12, 8), imagex_save_path=None):
        """
        Tests whether the show_image method works correctly on a specific test case.

        Args:
            imagex (np.ndarray): Image to display.
            titlex (str): Title of the window.
            figsizex (tuple): Figure size in inches (width, height).
            imagex_save_path (str, optional): If provided, saves the image to this path.

        Return:
            None
        """
        with patch("matplotlib.pyplot.figure") as mock_figure, \
            patch("matplotlib.pyplot.imshow") as mock_imshow, \
            patch("matplotlib.pyplot.title") as mock_title, \
            patch("matplotlib.pyplot.axis") as mock_axis, \
            patch("matplotlib.pyplot.savefig") as mock_savefig, \
            patch("matplotlib.pyplot.show") as mock_show:
            img.show_image(imagex, titlex, figsizex, imagex_save_path)

            _, kwargs_figure = mock_figure.call_args
            assert kwargs_figure["figsize"] == figsizex

            args_imshow, kwargs_imshow = mock_imshow.call_args
            assert np.array_equal(args_imshow[0], imagex)
            assert kwargs_imshow["cmap"] == "gray"

            args_title, _ = mock_title.call_args
            assert args_title[0] == titlex
            
            args_axis, _ = mock_axis.call_args
            assert args_axis[0] == "off"

            if imagex_save_path:
                args_savefig, kwargs_savefig = mock_savefig.call_args
                assert args_savefig[0] == imagex_save_path
                assert kwargs_savefig["bbox_inches"] == "tight"

                mock_savefig.assert_called_once()
                mock_show.assert_not_called()
            else:
                mock_show.assert_called_once_with()
                mock_savefig.assert_not_called()

            mock_figure.assert_called_once()
            mock_imshow.assert_called_once()
            mock_title.assert_called_once()
            mock_axis.assert_called_once()

            gc.collect()

    test_case_x(image1, "image1_save", (14, 10), image1_save_path)                                  #image1 save
    test_case_x(image1, "image1_show", (10, 6), None)                                               #image1 show
    test_case_x(image2, imagex_save_path=image2_save_path)                                          #image2 save
    test_case_x(image2)                                                                             #image2 show
    test_case_x(image3, figsizex=(18, 16), imagex_save_path=image3_save_path)                       #image3 save
    test_case_x(image3, figsizex=(7, 9), imagex_save_path=None)                                     #image3 show
    test_case_x(image4, titlex="image4_save", figsizex=(5, 7), imagex_save_path=image4_save_path)   #image4 save
    test_case_x(image4, titlex="image4_show")                                                       #image4 show
    test_case_x(image5, titlex="image5_save", imagex_save_path=image5_save_path)                    #image5 save
    test_case_x(image5, figsizex=(3, 3))                                                            #image5 show
    test_case_x(image6, "image6_save", figsizex=(2, 10), imagex_save_path=image6_save_path)         #image6 save
    test_case_x(image6, imagex_save_path=None)                                                      #image6 show


def test_preprocess_frame(init_tiff: tuple):
    """
    Tests whether the preprocess_frame method works correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file:
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Return:
        None
    """
    img, info = init_tiff
    f, c, h, w = info

    first_frame_channel_0 = img.arr[0, 0, :, :]
    middle_frame_channel_1 = img.arr[(f - 1) // 2, 1, :, :]
    last_frame_channel_2 = img.arr[f - 1, 2, :, :]

    kwargs1 = {"gauss": {}, "median": {}, "minmax": {}, "contrast": {}, "skip": []}
    kwargs2 = {
        "gauss": {"ksize": (3, 3), "sigmaX": 2.5},
        "median": {"ksize": 3},
        "minmax": {"alpha": 0, "beta": 255, "norm_type": cv2.NORM_MINMAX},
        "contrast": {"alpha": 1.5, "beta": 20},
        "skip": [],
    }
    kwargs3 = {
        "gauss": {"ksize": (1, 1)},
        "median": {"ksize": 9},
        "minmax": {},
        "contrast": {"alpha": 1.0},
        "skip": ["gauss", "median", "minmax", "contrast"],
    }
    kwargs4 = {
        "median": {"ksize": 7},
        "contrast": {"alpha": 0.5},
        "skip": ["gauss", "median"],
    }
    kwargs5 = {
        "gauss": {"sigmaX": 1.0},
        "minmax": {"alpha": 0, "beta": 1},
        "skip": ["minmax", "contrast"],
    }

    kwargs1_preprocess_first_frame = img.preprocess_frame(
        (first_frame_channel_0, kwargs1)
    )
    kwargs1_preprocess_middle_frame = img.preprocess_frame(
        (middle_frame_channel_1, kwargs1)
    )
    kwargs1_preprocess_last_frame = img.preprocess_frame(
        (last_frame_channel_2, kwargs1)
    )
    kwargs2_preprocess_first_frame = img.preprocess_frame(
        (first_frame_channel_0, kwargs2)
    )
    kwargs2_preprocess_middle_frame = img.preprocess_frame(
        (middle_frame_channel_1, kwargs2)
    )
    kwargs2_preprocess_last_frame = img.preprocess_frame(
        (last_frame_channel_2, kwargs2)
    )
    kwargs3_preprocess_first_frame = img.preprocess_frame(
        (first_frame_channel_0, kwargs3)
    )
    kwargs3_preprocess_middle_frame = img.preprocess_frame(
        (middle_frame_channel_1, kwargs3)
    )
    kwargs3_preprocess_last_frame = img.preprocess_frame(
        (last_frame_channel_2, kwargs3)
    )
    kwargs4_preprocess_first_frame = img.preprocess_frame(
        (first_frame_channel_0, kwargs4)
    )
    kwargs4_preprocess_middle_frame = img.preprocess_frame(
        (middle_frame_channel_1, kwargs4)
    )
    kwargs4_preprocess_last_frame = img.preprocess_frame(
        (last_frame_channel_2, kwargs4)
    )
    kwargs5_preprocess_first_frame = img.preprocess_frame(
        (first_frame_channel_0, kwargs5)
    )
    kwargs5_preprocess_middle_frame = img.preprocess_frame(
        (middle_frame_channel_1, kwargs5)
    )
    kwargs5_preprocess_last_frame = img.preprocess_frame(
        (last_frame_channel_2, kwargs5)
    )

    kwargs1_gauss = cv2.GaussianBlur(middle_frame_channel_1, (5, 5), 1.5)
    kwargs1_median = cv2.medianBlur(kwargs1_gauss, 5)
    kwargs1_minmax = cv2.normalize(kwargs1_median, None, 0, 255, cv2.NORM_MINMAX)
    kwargs1_contrast = cv2.convertScaleAbs(kwargs1_minmax, alpha=1.0, beta=0)
    assert np.array_equal(kwargs1_preprocess_middle_frame, kwargs1_contrast)

    kwargs2_gauss = cv2.GaussianBlur(first_frame_channel_0, (3, 3), 2.5)
    kwargs2_median = cv2.medianBlur(kwargs2_gauss, 3)
    kwargs2_minmax = cv2.normalize(kwargs2_median, None, 0, 255, cv2.NORM_MINMAX)
    kwargs2_contrast = cv2.convertScaleAbs(kwargs2_minmax, alpha=1.5, beta=20)
    assert np.array_equal(kwargs2_preprocess_first_frame, kwargs2_contrast)

    kwargs4_minmax = cv2.normalize(last_frame_channel_2, None, 0, 255, cv2.NORM_MINMAX)
    kwargs4_contrast = cv2.convertScaleAbs(kwargs4_minmax, alpha=0.5, beta=0)
    assert np.array_equal(kwargs4_preprocess_last_frame, kwargs4_contrast)

    kwargs5_gauss = cv2.GaussianBlur(first_frame_channel_0, (5, 5), 1.0)
    kwargs5_median = cv2.medianBlur(kwargs5_gauss, 5)
    assert np.array_equal(kwargs5_preprocess_first_frame, kwargs5_median)

    assert not np.array_equal(kwargs1_preprocess_first_frame, first_frame_channel_0)
    assert not np.array_equal(kwargs1_preprocess_middle_frame, middle_frame_channel_1)
    assert not np.array_equal(kwargs1_preprocess_last_frame, last_frame_channel_2)
    assert not np.array_equal(kwargs2_preprocess_first_frame, first_frame_channel_0)
    assert not np.array_equal(kwargs2_preprocess_middle_frame, middle_frame_channel_1)
    assert not np.array_equal(kwargs2_preprocess_last_frame, last_frame_channel_2)
    assert np.array_equal(kwargs3_preprocess_first_frame, first_frame_channel_0)
    assert np.array_equal(kwargs3_preprocess_middle_frame, middle_frame_channel_1)
    assert np.array_equal(kwargs3_preprocess_last_frame, last_frame_channel_2)
    assert not np.array_equal(kwargs4_preprocess_first_frame, first_frame_channel_0)
    assert not np.array_equal(kwargs4_preprocess_middle_frame, middle_frame_channel_1)
    assert not np.array_equal(kwargs4_preprocess_last_frame, last_frame_channel_2)
    assert not np.array_equal(kwargs5_preprocess_first_frame, first_frame_channel_0)
    assert not np.array_equal(kwargs5_preprocess_middle_frame, middle_frame_channel_1)
    assert not np.array_equal(kwargs5_preprocess_last_frame, last_frame_channel_2)

    assert isinstance(kwargs1_preprocess_first_frame, np.ndarray)
    assert isinstance(kwargs1_preprocess_middle_frame, np.ndarray)
    assert isinstance(kwargs1_preprocess_last_frame, np.ndarray)
    assert isinstance(kwargs2_preprocess_first_frame, np.ndarray)
    assert isinstance(kwargs2_preprocess_middle_frame, np.ndarray)
    assert isinstance(kwargs2_preprocess_last_frame, np.ndarray)
    assert isinstance(kwargs3_preprocess_first_frame, np.ndarray)
    assert isinstance(kwargs3_preprocess_middle_frame, np.ndarray)
    assert isinstance(kwargs3_preprocess_last_frame, np.ndarray)
    assert isinstance(kwargs4_preprocess_first_frame, np.ndarray)
    assert isinstance(kwargs4_preprocess_middle_frame, np.ndarray)
    assert isinstance(kwargs4_preprocess_last_frame, np.ndarray)
    assert isinstance(kwargs5_preprocess_first_frame, np.ndarray)
    assert isinstance(kwargs5_preprocess_middle_frame, np.ndarray)
    assert isinstance(kwargs5_preprocess_last_frame, np.ndarray)

    assert first_frame_channel_0.shape == (h, w)
    assert middle_frame_channel_1.shape == (h, w)
    assert last_frame_channel_2.shape == (h, w)
    assert kwargs1_preprocess_first_frame.shape == first_frame_channel_0.shape
    assert kwargs1_preprocess_middle_frame.shape == middle_frame_channel_1.shape
    assert kwargs1_preprocess_last_frame.shape == last_frame_channel_2.shape
    assert kwargs2_preprocess_first_frame.shape == first_frame_channel_0.shape
    assert kwargs2_preprocess_middle_frame.shape == middle_frame_channel_1.shape
    assert kwargs2_preprocess_last_frame.shape == last_frame_channel_2.shape
    assert kwargs3_preprocess_first_frame.shape == first_frame_channel_0.shape
    assert kwargs3_preprocess_middle_frame.shape == middle_frame_channel_1.shape
    assert kwargs3_preprocess_last_frame.shape == last_frame_channel_2.shape
    assert kwargs4_preprocess_first_frame.shape == first_frame_channel_0.shape
    assert kwargs4_preprocess_middle_frame.shape == middle_frame_channel_1.shape
    assert kwargs4_preprocess_last_frame.shape == last_frame_channel_2.shape
    assert kwargs5_preprocess_first_frame.shape == first_frame_channel_0.shape
    assert kwargs5_preprocess_middle_frame.shape == middle_frame_channel_1.shape
    assert kwargs5_preprocess_last_frame.shape == last_frame_channel_2.shape


def test_preprocess_stack(init_tiff: tuple):
    """
    Tests whether the preprocess_stack method works correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file:
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Return:
        None
    """
    img, info = init_tiff
    f, c, h, w = info
    tiff_arr = img.arr

    stack1 = np.asarray(tiff_arr[:, 0, :, :])
    stack2 = np.asarray([tiff_arr[0, 1, :, :]])
    stack3 = np.asarray(
        [tiff_arr[0, 2, :, :], tiff_arr[(f - 1) // 2, 1, :, :], tiff_arr[f - 1, 0, :, :]]
    )
    stack4 = np.asarray(tiff_arr[: (f - 1) // 2, 2, :, :])
    stack5 = np.asarray(tiff_arr[:, 1, :, :])
    stack6 = np.asarray(tiff_arr[:, 2, :, :])
    stack7 = tiff_arr

    kwargs6 = {
        "gauss": {"ksize": (3, 3), "sigmaX": 2.5},
        "median": {"ksize": 3},
        "minmax": {"alpha": 0, "beta": 255, "norm_type": cv2.NORM_MINMAX},
        "contrast": {"alpha": 1.5, "beta": 20},
        "skip": [],
    }
    kwargs7 = {
        "gauss": {"ksize": (7, 7)},
        "median": {"ksize": 9},
        "minmax": {},
        "contrast": {"alpha": 1.0},
        "skip": ["gauss", "median", "minmax", "contrast"],
    }
    kwargs8 = {
        "gauss": {"sigmaX": 1.0},
        "minmax": {"alpha": 0, "beta": 1},
        "skip": ["gauss", "median"],
    }
    kwargs9 = {}

    def test_case_x(stackx: np.ndarray, **kwargsx):
        """
        Tests whether the preprocess_stack method works correctly on a specific test case.

        Args:
            stackx (np.ndarray): Input stack of frames (shape: N x H x W).
            **kwargsx: Dictionary with preprocessing parameters:
                - gauss (dict): {'ksize': (int, int), 'sigmaX': float}
                - median (dict): {'ksize': int}
                - normalize (dict): {'alpha': int, 'beta': int, 'norm_type': int}
                - contrast (dict): {'alpha': float, 'beta': int}
                - skip (list[str]): steps to skip (e.g., ['gauss', 'median'])

        Returns:
            None
        """
        with patch("src.cell_tracking.tiffclass.Pool") as mock_pool:
            mock_pool_instance = mock_pool.return_value.__enter__.return_value
            mock_pool_instance.map.side_effect = lambda func, arr1: [np.zeros_like(x[0]) for x in arr1]

            frames = [(stackx[i], kwargsx) for i in range(stackx.shape[0])]

            result = img.preprocess_stack(stackx, **kwargsx)

            pool_args, _ = mock_pool.call_args
            assert pool_args[0] == cpu_count()

            pool_map_args, _ = mock_pool_instance.map.call_args
            assert callable(pool_map_args[0])
            for i in range(0, len(frames)):
                assert np.array_equal(pool_map_args[1][i][0], frames[i][0])
                assert pool_map_args[1][i][1] == frames[i][1]

            assert result.shape == stackx.shape
            assert isinstance(result, np.ndarray)
            mock_pool.assert_called_once()
            mock_pool_instance.map.assert_called_once()

            del result, frames
            gc.collect()

    test_case_x(stack1, **kwargs6)
    test_case_x(stack2, **kwargs6)
    test_case_x(stack3, **kwargs6)
    test_case_x(stack4, **kwargs6)
    test_case_x(stack5, **kwargs6)
    test_case_x(stack6, **kwargs6)
    test_case_x(stack7, **kwargs6)
    test_case_x(stack1, **kwargs7)
    test_case_x(stack2, **kwargs7)
    test_case_x(stack3, **kwargs7)
    test_case_x(stack4, **kwargs7)
    test_case_x(stack5, **kwargs7)
    test_case_x(stack6, **kwargs7)
    test_case_x(stack7, **kwargs7)
    test_case_x(stack1, **kwargs8)
    test_case_x(stack2, **kwargs8)
    test_case_x(stack3, **kwargs8)
    test_case_x(stack4, **kwargs8)
    test_case_x(stack5, **kwargs8)
    test_case_x(stack6, **kwargs8)
    test_case_x(stack7, **kwargs8)
    test_case_x(stack1, **kwargs9)
    test_case_x(stack2, **kwargs9)
    test_case_x(stack3, **kwargs9)
    test_case_x(stack4, **kwargs9)
    test_case_x(stack5, **kwargs9)
    test_case_x(stack6, **kwargs9)
    test_case_x(stack7, **kwargs9)