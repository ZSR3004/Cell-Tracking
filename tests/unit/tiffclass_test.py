import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
import cv2, pytest
from src import tiffclass as tiff
import numpy as np

@pytest.fixture
def sample_tiff():
    path = "../../datasets/nuclei_labeled/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif"
    f, c, h, w = 96, 3, 520, 2329
    return path, f, c, h, w

def hi_init(sample_tiff):
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


def hi_isolate_channel(sample_tiff):
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

def hi_save_original_video(sample_tiff, tmp_path):
    """
    Tests whether the save_original_video method works correctly.

    Args:
        sample_tiff (tuple): A tuple containing information about the TIFF file:
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
        tmp_path (pathlib.Path): A path to a temporary directory (this is a fixture in Pytest).

    Return:
    """
    path, f, c, h, w = sample_tiff

    #finish this docstring
    #Note: i'm gonna use tmp_path by calling save_original_video and having it save the video to tmp_path
    #Edit my test cases bc we edited Ziyad's function

    #NOW RUN ALL THESE TESTS!
    #Add to func below: "Note that running these tests will cause x windows to pop up...."

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
    
def test_show_image(sample_tiff, tmp_path):
    """
    Tests whether the show_image method works correctly. 

    Args:
        sample_tiff (tuple): A tuple containing information about the TIFF file:
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
        tmp_path (pathlib.Path): A path to a temporary directory (this is a fixture in Pytest).

    Return:
        None
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)

    kwargs1 = {"gauss": {"ksize": (3, 3), "sigmaX": 1.5}, "median": {"ksize": 3}, "minmax": {"alpha": 0, "beta": 255, "norm_type": cv2.NORM_MINMAX}, "contrast": {"alpha": 1.5, "beta": 20}, "skip": []}
    kwargs2 = {"gauss": {"ksize": (7, 7)}, "median": {"ksize": 9}, "minmax": {}, "contrast": {"alpha": 1.0}, "skip": ["gauss", "median", "minmax", "contrast"]}
    kwargs3 = {"gauss": {"sigmaX": 1.5}, "minmax": {"alpha": 0, "beta": 1}, "skip": ["minmax", "contrast"]}

    image1 = img.arr[0, 2, :, :]
    image2 = img.arr[(f-1)//2, 0, :, :]
    image3 = img.arr[f-1, 1, :, :]
    image4 = img.preprocess_frame((img.arr[0, 1, :, :], kwargs1))
    image5 = img.preprocess_frame((img.arr[(f-1)//2, 2, :, :], kwargs2))
    image6 = img.preprocess_frame((img.arr[f-1, 0, :, :], kwargs3))

    image1_save_path = tmp_path / 'image1_save.png'
    image2_save_path = tmp_path / 'image2_save.png'
    image3_save_path = tmp_path / 'image3_save.png'
    image4_save_path = tmp_path / 'image4_save.png'
    image5_save_path = tmp_path / 'image5_save.png'
    image6_save_path = tmp_path / 'image6_save.png'

    image1_save = img.show_image(image1, "image1_save", (14, 10), image1_save_path)
    image1_show = img.show_image(image1, "image1_show", (10, 6), None)
    image2_save = img.show_image(image2, title="image2_save", save_path=image2_save_path)
    image2_show = img.show_image(image2, title="image2_show")
    image3_save = img.show_image(image3, title="image3_save", figsize=(18, 16), save_path=image3_save_path)
    image3_show = img.show_image(image3, title="image3_show", figsize=(7, 9), save_path=None)
    image4_save = img.show_image(image4, title="image4_save", figsize=(5,7), save_path=image4_save_path)
    image4_show = img.show_image(image4, title="image4_show")
    image5_save = img.show_image(image5, title="image5_save", save_path=image5_save_path)
    image5_show = img.show_image(image5, title="image5_show",figsize=(3,3))
    image6_save = img.show_image(image6, "image6_save", figsize=(2,10), save_path=image6_save_path)
    image6_show = img.show_image(image6, title="image6_show", save_path=None)

    """
    REVERT BACK:
    image1_save = img.show_image(image1, "image1_save", (14, 10), image1_save_path)
    image1_show = img.show_image(image1, "image1_show", (10, 6), None)
    image2_save = img.show_image(image2, save_path=image2_save_path)
    image2_show = img.show_image(image2)
    image3_save = img.show_image(image3, figsize=(18, 16), save_path=image3_save_path)
    image3_show = img.show_image(image3, figsize=(7, 9), save_path=None)
    image4_save = img.show_image(image4, title="image4_save", figsize=(5,7), save_path=image4_save_path)
    image4_show = img.show_image(image4, title="image4_show")
    image5_save = img.show_image(image5, title="image5_save", save_path=image5_save_path)
    image5_show = img.show_image(image5, figsize=(3,3))
    image6_save = img.show_image(image6, "image6_save", figsize=(2,10), save_path=image6_save_path)
    image6_show = img.show_image(image6, save_path=None)
    """

    assert image1_save_path.exists()
    assert image2_save_path.exists()
    assert image3_save_path.exists()
    assert image4_save_path.exists()
    assert image5_save_path.exists()
    assert image6_save_path.exists()

    assert os.path.getsize(image1_save_path) > 0
    assert os.path.getsize(image2_save_path) > 0
    assert os.path.getsize(image3_save_path) > 0
    assert os.path.getsize(image4_save_path) > 0
    assert os.path.getsize(image5_save_path) > 0
    assert os.path.getsize(image6_save_path) > 0

def hi_preprocess_frame(sample_tiff):
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

    first_frame_channel_0 = img.arr[0, 0, :, :]
    middle_frame_channel_1 = img.arr[(f-1)//2, 1, :, :]
    last_frame_channel_2 = img.arr[f-1, 2, :, :]

    kwargs1 = {"gauss": {}, "median": {}, "minmax": {}, "contrast": {}, "skip": []}
    kwargs2 = {"gauss": {"ksize": (3, 3), "sigmaX": 2.5}, "median": {"ksize": 3}, "minmax": {"alpha": 0, "beta": 255, "norm_type": cv2.NORM_MINMAX}, "contrast": {"alpha": 1.5, "beta": 20}, "skip": []}
    kwargs3 = {"gauss": {"ksize": (1, 1)}, "median": {"ksize": 9}, "minmax": {}, "contrast": {"alpha": 1.0}, "skip": ["gauss", "median", "minmax", "contrast"]}
    kwargs4 = {"median": {"ksize": 7}, "contrast": {"alpha": 0.5}, "skip": ["gauss", "median"]}
    kwargs5 = {"gauss": {"sigmaX": 1.0}, "minmax": {"alpha": 0, "beta": 1}, "skip": ["minmax", "contrast"]}

    kwargs1_preprocess_first_frame = img.preprocess_frame((first_frame_channel_0, kwargs1))
    kwargs1_preprocess_middle_frame = img.preprocess_frame((middle_frame_channel_1, kwargs1))
    kwargs1_preprocess_last_frame = img.preprocess_frame((last_frame_channel_2, kwargs1))
    kwargs2_preprocess_first_frame = img.preprocess_frame((first_frame_channel_0, kwargs2))
    kwargs2_preprocess_middle_frame = img.preprocess_frame((middle_frame_channel_1, kwargs2))
    kwargs2_preprocess_last_frame = img.preprocess_frame((last_frame_channel_2, kwargs2))
    kwargs3_preprocess_first_frame = img.preprocess_frame((first_frame_channel_0, kwargs3))
    kwargs3_preprocess_middle_frame = img.preprocess_frame((middle_frame_channel_1, kwargs3))
    kwargs3_preprocess_last_frame = img.preprocess_frame((last_frame_channel_2, kwargs3))
    kwargs4_preprocess_first_frame = img.preprocess_frame((first_frame_channel_0, kwargs4))
    kwargs4_preprocess_middle_frame = img.preprocess_frame((middle_frame_channel_1, kwargs4))
    kwargs4_preprocess_last_frame = img.preprocess_frame((last_frame_channel_2, kwargs4))
    kwargs5_preprocess_first_frame = img.preprocess_frame((first_frame_channel_0, kwargs5))
    kwargs5_preprocess_middle_frame = img.preprocess_frame((middle_frame_channel_1, kwargs5))
    kwargs5_preprocess_last_frame = img.preprocess_frame((last_frame_channel_2, kwargs5))

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

def hi_preprocess_stack(sample_tiff):
    """
    Tests whether the preprocess_stack method works correctly.

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

    stack1 = np.asarray(img.arr[:, 0, :, :])
    stack2 = np.asarray([img.arr[0, 1, :, :]])
    stack3 = np.asarray([img.arr[0, 2, :, :], img.arr[(f-1)//2, 1, :, :], img.arr[f-1, 0, :, :]])
    stack4 = np.asarray(img.arr[:(f-1)//2, 2, :, :])
    stack5 = np.asarray(img.arr[:, 1, :, :])
    stack6 = np.asarray(img.arr[:, 2, :, :])

    kwargs6 = {"gauss": {"ksize": (3, 3), "sigmaX": 2.5}, "median": {"ksize": 3}, "minmax": {"alpha": 0, "beta": 255, "norm_type": cv2.NORM_MINMAX}, "contrast": {"alpha": 1.5, "beta": 20}, "skip": []}
    kwargs7 = {"gauss": {"ksize": (7, 7)}, "median": {"ksize": 9}, "minmax": {}, "contrast": {"alpha": 1.0}, "skip": ["gauss", "median", "minmax", "contrast"]}
    kwargs8 = {"gauss": {"sigmaX": 1.0}, "minmax": {"alpha": 0, "beta": 1}, "skip": ["gauss", "median"]}

    kwargs6_preprocess_stack1 = img.preprocess_stack(stack1, **kwargs6)
    kwargs6_preprocess_stack2 = img.preprocess_stack(stack2, **kwargs6)
    kwargs6_preprocess_stack3 = img.preprocess_stack(stack3, **kwargs6)
    kwargs6_preprocess_stack4 = img.preprocess_stack(stack4, **kwargs6)
    kwargs6_preprocess_stack5 = img.preprocess_stack(stack5, **kwargs6)
    kwargs6_preprocess_stack6 = img.preprocess_stack(stack6, **kwargs6)
    kwargs7_preprocess_stack1 = img.preprocess_stack(stack1, **kwargs7)
    kwargs7_preprocess_stack2 = img.preprocess_stack(stack2, **kwargs7)
    kwargs7_preprocess_stack3 = img.preprocess_stack(stack3, **kwargs7)
    kwargs7_preprocess_stack4 = img.preprocess_stack(stack4, **kwargs7)
    kwargs7_preprocess_stack5 = img.preprocess_stack(stack5, **kwargs7)
    kwargs7_preprocess_stack6 = img.preprocess_stack(stack6, **kwargs7)
    kwargs8_preprocess_stack1 = img.preprocess_stack(stack1, **kwargs8)
    kwargs8_preprocess_stack2 = img.preprocess_stack(stack2, **kwargs8)
    kwargs8_preprocess_stack3 = img.preprocess_stack(stack3, **kwargs8)
    kwargs8_preprocess_stack4 = img.preprocess_stack(stack4, **kwargs8)
    kwargs8_preprocess_stack5 = img.preprocess_stack(stack5, **kwargs8)
    kwargs8_preprocess_stack6 = img.preprocess_stack(stack6, **kwargs8)

    assert np.array_equal(np.asarray([img.preprocess_frame((np.asarray(img.arr[0, 1, :, :]), kwargs6))]), kwargs6_preprocess_stack2)
    assert np.array_equal(np.asarray([img.preprocess_frame((np.asarray(img.arr[0, 1, :, :]), kwargs7))]), kwargs7_preprocess_stack2)
    assert np.array_equal(np.asarray([img.preprocess_frame((np.asarray(img.arr[0, 1, :, :]), kwargs8))]), kwargs8_preprocess_stack2)

    assert np.array_equal(np.asarray([img.preprocess_frame((np.asarray(img.arr[0, 2, :, :]), kwargs6)), img.preprocess_frame((np.asarray(img.arr[(f-1)//2, 1, :, :]), kwargs6)), img.preprocess_frame((np.asarray(img.arr[f-1, 0, :, :]), kwargs6))]), kwargs6_preprocess_stack3)
    assert np.array_equal(np.asarray([img.preprocess_frame((np.asarray(img.arr[0, 2, :, :]), kwargs7)), img.preprocess_frame((np.asarray(img.arr[(f-1)//2, 1, :, :]), kwargs7)), img.preprocess_frame((np.asarray(img.arr[f-1, 0, :, :]), kwargs7))]), kwargs7_preprocess_stack3)
    assert np.array_equal(np.asarray([img.preprocess_frame((np.asarray(img.arr[0, 2, :, :]), kwargs8)), img.preprocess_frame((np.asarray(img.arr[(f-1)//2, 1, :, :]), kwargs8)), img.preprocess_frame((np.asarray(img.arr[f-1, 0, :, :]), kwargs8))]), kwargs8_preprocess_stack3)

    assert not np.array_equal(kwargs6_preprocess_stack1, stack1)
    assert not np.array_equal(kwargs6_preprocess_stack2, stack2)
    assert not np.array_equal(kwargs6_preprocess_stack3, stack3)
    assert not np.array_equal(kwargs6_preprocess_stack4, stack4)
    assert not np.array_equal(kwargs6_preprocess_stack5, stack5)
    assert not np.array_equal(kwargs6_preprocess_stack6, stack6)
    assert np.array_equal(kwargs7_preprocess_stack1, stack1)
    assert np.array_equal(kwargs7_preprocess_stack2, stack2)
    assert np.array_equal(kwargs7_preprocess_stack3, stack3)
    assert np.array_equal(kwargs7_preprocess_stack4, stack4)
    assert np.array_equal(kwargs7_preprocess_stack5, stack5)
    assert np.array_equal(kwargs7_preprocess_stack6, stack6)
    assert not np.array_equal(kwargs8_preprocess_stack1, stack1)
    assert not np.array_equal(kwargs8_preprocess_stack2, stack2)
    assert not np.array_equal(kwargs8_preprocess_stack3, stack3)
    assert not np.array_equal(kwargs8_preprocess_stack4, stack4)
    assert not np.array_equal(kwargs8_preprocess_stack5, stack5)
    assert not np.array_equal(kwargs8_preprocess_stack6, stack6)

    assert isinstance(kwargs6_preprocess_stack1, np.ndarray)
    assert isinstance(kwargs6_preprocess_stack2, np.ndarray)
    assert isinstance(kwargs6_preprocess_stack3, np.ndarray)
    assert isinstance(kwargs6_preprocess_stack4, np.ndarray)
    assert isinstance(kwargs6_preprocess_stack5, np.ndarray)
    assert isinstance(kwargs6_preprocess_stack6, np.ndarray)
    assert isinstance(kwargs7_preprocess_stack1, np.ndarray)
    assert isinstance(kwargs7_preprocess_stack2, np.ndarray)
    assert isinstance(kwargs7_preprocess_stack3, np.ndarray)
    assert isinstance(kwargs7_preprocess_stack4, np.ndarray)
    assert isinstance(kwargs7_preprocess_stack5, np.ndarray)
    assert isinstance(kwargs7_preprocess_stack6, np.ndarray)
    assert isinstance(kwargs8_preprocess_stack1, np.ndarray)
    assert isinstance(kwargs8_preprocess_stack2, np.ndarray)
    assert isinstance(kwargs8_preprocess_stack3, np.ndarray)
    assert isinstance(kwargs8_preprocess_stack4, np.ndarray)
    assert isinstance(kwargs8_preprocess_stack5, np.ndarray)
    assert isinstance(kwargs8_preprocess_stack6, np.ndarray)

    assert kwargs6_preprocess_stack1.shape == stack1.shape
    assert kwargs6_preprocess_stack2.shape == stack2.shape
    assert kwargs6_preprocess_stack3.shape == stack3.shape
    assert kwargs6_preprocess_stack4.shape == stack4.shape
    assert kwargs6_preprocess_stack5.shape == stack5.shape
    assert kwargs6_preprocess_stack6.shape == stack6.shape
    assert kwargs7_preprocess_stack1.shape == stack1.shape
    assert kwargs7_preprocess_stack2.shape == stack2.shape
    assert kwargs7_preprocess_stack3.shape == stack3.shape
    assert kwargs7_preprocess_stack4.shape == stack4.shape
    assert kwargs7_preprocess_stack5.shape == stack5.shape
    assert kwargs7_preprocess_stack6.shape == stack6.shape
    assert kwargs8_preprocess_stack1.shape == stack1.shape
    assert kwargs8_preprocess_stack2.shape == stack2.shape
    assert kwargs8_preprocess_stack3.shape == stack3.shape
    assert kwargs8_preprocess_stack4.shape == stack4.shape
    assert kwargs8_preprocess_stack5.shape == stack5.shape
    assert kwargs8_preprocess_stack6.shape == stack6.shape