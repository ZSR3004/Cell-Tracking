import sys, os, cv2, pytest
from src import tiffclass as tiff
import numpy as np

@pytest.fixture
def sample_tiff():
    path = "../../datasets/nuclei_labeled/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif"
    f, c, h, w = 96, 3, 520, 2329
    return path, f, c, h, w

def test_init(sample_tiff):
    """
    Tests whether the Tiff class initializes correctly.

    Args:
        sample_tiff (tuple): A tuple containing information about the TIFF file.
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Return:
        None
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)

    assert img.path == str(path)
    assert isinstance(img.arr, np.ndarray)
    assert hasattr(img, "path")
    assert hasattr(img, "timestamp")
    assert hasattr(img, "arr")

    assert img.arr.shape == (f,c,h,w)
    assert img.arr.shape[0] == f  # number of frames
    assert img.arr.shape[1] == c  # number of channels
    assert img.arr.shape[2] == h  # height
    assert img.arr.shape[3] == w  # width


def test_isolate_channel(sample_tiff):
    """
    Tests whether the isolate_channel method works correctly.

    Args:
        sample_tiff (tuple): A tuple containing information about the TIFF file.
            - path (str): The path to the TIFF file:
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Return:
        None
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)

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

def test_save_original_video(sample_tiff, tmp_path):
    """
    Tests whether the save_original_video method works correctly.

    Args:

        tmp_path: A path to a temporary directory (this is a fixture in Pytest).

    Return:
    """
    path, f, c, h, w = sample_tiff

    #finish this docstring
    #Note: i'm gonna use tmp_path by calling save_original_video and having it save the video to tmp_path
    #finish once i ask Ziyad (who wrote this function) why we don't assert that im, image_stack, ax, and fig are not None. Because it breaks if either
    #im, image_stack, ax, or fig are None

    #what to assert: check if exists, check if not empty
    """
    also do if im, image_stack, ax, fig, T, or fps are None:
        - case: im, image_stack, ax, fig, T, and fps are all not None
        - case: im, image_stack, ax, fig, T, and fps are all None
            - image_stack is (T, H, W)
        - case: im, image_stack, fig, and fps are None. ax and T are not None.
        - case: ax and T are None. im, image_stack, fig, and fps are not None.
            - image_stack is (T, H, W, 3)
    """

    raise NotImplemented
    
def test_show_image():
    """
    Tests whether the show_image method works correctly.

    Args:

    Return:
        None
    """

    #finish this docstring
    #test both cases: if save_path is a str, and if it's None
    #if save_path is a str, test if it's saved. If save_path is None, test if it's shown.

    raise NotImplemented

def test_preprocess_frame(sample_tiff):
    """
    Tests whether the preprocess_frame method works correctly.

    Args:
        sample_tiff (tuple): A tuple containing information about the TIFF file:
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Return:
        None
    """

    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)

    first_frame = img.arr[0]
    middle_frame = img.arr[(f-1)//2]

    kwargs1 = {"gauss": {}, "median": {}, "minmax": {}, "contrast": {}, "skip": []}
    kwargs2 = {"gauss": {"ksize": (8, 8), "sigmaX": 2.5}, "median": {"ksize": 4}, "minmax": {"alpha": 50, "beta": 200, "norm_type": cv2.NORM_MINMAX}, "contrast": {"alpha": 1.5, "beta": 20}, "skip": []}
    kwargs3 = {"gauss": {"ksize": (4, 4)}, "median": {"ksize": 10}, "minmax": {}, "contrast": {"alpha": 1.0}, "skip": ["gauss", "median", "minmax", "contrast"]}
    kwargs4 = {"median": {"ksize": 7}, "contrast": {"alpha": 0.5}, "skip": ["gauss", "median"]}
    kwargs5 = {"gauss": {"sigmaX": 1.0}, "minmax": {"alpha": 0, "beta": 1}, "skip": ["minmax", "contrast"]}

    kwargs1_preprocess_first_frame = img.preprocess_frame((first_frame, kwargs1))
    kwargs1_preprocess_middle_frame = img.preprocess_frame((middle_frame, kwargs1))
    kwargs2_preprocess_first_frame = img.preprocess_frame((first_frame, kwargs2))
    kwargs2_preprocess_middle_frame = img.preprocess_frame((middle_frame, kwargs2))
    kwargs3_preprocess_first_frame = img.preprocess_frame((first_frame, kwargs3))
    kwargs3_preprocess_middle_frame = img.preprocess_frame((middle_frame, kwargs3))
    kwargs4_preprocess_first_frame = img.preprocess_frame((first_frame, kwargs4))
    kwargs4_preprocess_middle_frame = img.preprocess_frame((middle_frame, kwargs4))
    kwargs5_preprocess_first_frame = img.preprocess_frame((first_frame, kwargs5))
    kwargs5_preprocess_middle_frame = img.preprocess_frame((middle_frame, kwargs5))

    kwargs1_gauss = cv2.GaussianBlur(middle_frame, (5, 5), 1.5)
    kwargs1_median = cv2.medianBlur(kwargs1_gauss, 5)
    kwargs1_minmax = cv2.normalize(kwargs1_median, None, 0, 255, cv2.NORM_MINMAX)
    kwargs1_contrast = cv2.convertScaleAbs(kwargs1_minmax, alpha=1.0, beta=0)
    assert kwargs1_preprocess_middle_frame == kwargs1_contrast

    kwargs2_gauss = cv2.GaussianBlur(first_frame, (8, 8), 2.5)
    kwargs2_median = cv2.medianBlur(kwargs2_gauss, 4)
    kwargs2_minmax = cv2.normalize(kwargs2_median, None, 50, 200, cv2.NORM_MINMAX)
    kwargs2_contrast = cv2.convertScaleAbs(kwargs2_minmax, alpha=1.5, beta=20)
    assert kwargs2_preprocess_first_frame == kwargs2_contrast

    kwargs4_minmax = cv2.normalize(middle_frame, None, 0, 255, cv2.NORM_MINMAX)
    kwargs4_contrast = cv2.convertScaleAbs(kwargs4_minmax, alpha=0.5, beta=0)
    assert kwargs4_preprocess_middle_frame == kwargs4_contrast

    kwargs5_gauss = cv2.GaussianBlur(first_frame, (5, 5), 1.0)
    kwargs5_median = cv2.medianBlur(kwargs5_gauss, 5)
    assert kwargs5_preprocess_first_frame == kwargs5_median

    assert kwargs1_preprocess_first_frame != first_frame
    assert kwargs1_preprocess_middle_frame != middle_frame
    assert kwargs2_preprocess_first_frame != first_frame
    assert kwargs2_preprocess_middle_frame != middle_frame
    assert kwargs3_preprocess_first_frame == first_frame
    assert kwargs3_preprocess_middle_frame == middle_frame
    assert kwargs4_preprocess_first_frame != first_frame
    assert kwargs4_preprocess_middle_frame != middle_frame
    assert kwargs5_preprocess_first_frame != first_frame
    assert kwargs5_preprocess_middle_frame != middle_frame

    assert isinstance(kwargs1_preprocess_first_frame, np.ndarray)
    assert isinstance(kwargs1_preprocess_middle_frame, np.ndarray)
    assert isinstance(kwargs2_preprocess_first_frame, np.ndarray)
    assert isinstance(kwargs2_preprocess_middle_frame, np.ndarray)
    assert isinstance(kwargs3_preprocess_first_frame, np.ndarray)
    assert isinstance(kwargs3_preprocess_middle_frame, np.ndarray)
    assert isinstance(kwargs4_preprocess_first_frame, np.ndarray)
    assert isinstance(kwargs4_preprocess_middle_frame, np.ndarray)
    assert isinstance(kwargs5_preprocess_first_frame, np.ndarray)
    assert isinstance(kwargs5_preprocess_middle_frame, np.ndarray)
    
    assert kwargs1_preprocess_first_frame.shape == (f, h, w)
    assert kwargs1_preprocess_middle_frame.shape == (f, h, w)
    assert kwargs2_preprocess_first_frame.shape == (f, h, w)
    assert kwargs2_preprocess_middle_frame.shape == (f, h, w)
    assert kwargs3_preprocess_first_frame.shape == (f, h, w)
    assert kwargs3_preprocess_middle_frame.shape == (f, h, w)
    assert kwargs4_preprocess_first_frame.shape == (f, h, w)
    assert kwargs4_preprocess_middle_frame.shape == (f, h, w)
    assert kwargs5_preprocess_first_frame.shape == (f, h, w)
    assert kwargs5_preprocess_middle_frame.shape == (f, h, w)

def test_preprocess_stack():

    raise NotImplemented
    #THEN RUN THESE TESTS!