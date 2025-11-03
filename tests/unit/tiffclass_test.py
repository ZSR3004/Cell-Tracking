import sys, os, pytest
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
    middle_frame = img.arr[f-1//2]
    last_frame = img.arr[f-1]


    """
    cases to test:
        - skip == [] (this accounts for all cases of "gauss", "median", "minmax", and "contrast" not being in skip)
            - "gauss" in kwargs
            - "median" in kwargs
            - "minmax" in kwargs
            - "contrast" in kwargs
        - skip == ["gauss", "median", "minmax", "contrast"] (this accounts for all cases of "gauss", "median", "minmax", and "contrast" being in skip)
            - doesn't matter what's in kwargs
        - skip == ["gauss", "median"] (this accounts for just a few things in skip, and different things in kwargs and different things not in kwargs)
            - "gauss" not in kwargs
            - "median" in kwargs
            - "minmax" not in kwargs
            - "contrast" in kwargs
        - skip == ["minmax", "contrast"] (this accounts for just a few things in skip, and different things in kwargs and different things not in kwargs)
            - "gauss" in kwargs
            - "median" not in kwargs
            - "minmax" in kwargs
            - "contrast" not in kwargs
    """

    raise NotImplemented


#ADD MORE!!

