import sys, os, pytest
from sympy import Idx
from src import optical_flow as flow
import numpy as np
from src import tiffclass as tiff
import matplotlib
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
import matplotlib.pyplot as plt

@pytest.fixture
def sample_tiff():
    path = "../../datasets/nuclei_labeled/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif"
    f, c, h, w = 96, 3, 520, 2329
    return path, f, c, h, w

def test_combine_flows(sample_tiff):
    """
    Tests whether the combine_flows function works correctly.
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)

    channel_0 = img.isolate_channel(0)
    channel_1 = img.isolate_channel(1)
    channel_2 = img.isolate_channel(2)

    combine_0_1 = img.combine_flows([channel_0,channel_1])
    combine_0_2 = img.combine_flows([channel_0,channel_2])
    combine_1_2 = img.combine_flows([channel_1,channel_2])

    assert combine_0_1.shape == (f, h, w)
    assert combine_0_2.shape == (f, h, w)
    assert combine_1_2 == (f, h, w)

    np.testing.assert_array_equal(combine_0_1, channel_0 + channel_1)
    np.testing.assert_array_equal(combine_0_2, channel_0 + channel_2)
    np.testing.assert_array_equal(combine_1_2, channel_1 + channel_2)


def test_compute_flow_pair(sample_tiff):
    """
    Tests whether the compute_flow_pair function works correctly.
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)


    # Grab the first two frames of channel 0
    f1 = img.arr[0, 0]  # shape: (height, width)
    f2 = img.arr[1, 0]  # shape: (height, width)

    flow_args = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0
    }

    args = (f1, f2, flow_args)
    my_flow = flow.compute_flow_pair(args)

    # Test output type
    assert isinstance(my_flow, np.ndarray)

    # Test output shape
    assert my_flow.shape == (h, w, 2)

def test_optical_flow(sample_tiff):
    """
    Tests whether the optical_flow function works correctly.
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)


    # isolate channel for testing
    channel = img.isolate_channel(0)

    #arguments for testing
    flow_args = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0
    }

    my_flow = flow.optical_flow(channel, flow_args)

    # Test output type
    assert isinstance(my_flow, np.ndarray)

    # Test output shape
    assert my_flow.shape == (f-1, h, w, 2)

def test_calculate_optical_flow(sample_tiff):
    """
    Tests whether the calculate_optical_flow function works correctly.
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)

    
    # Example preprocessing: normalize frames to 0-1, apply small Gaussian blur
    process_args = {
        "normalize": True,
        "gaussian_blur": 3
    }

    #arguments for testing
    flow_args = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0
    }

    my_flow = flow.calculate_optical_flow(img.arr, process_args, flow_args, False)

    # Test output type
    assert isinstance(my_flow, np.ndarray)

    # Test output shape
    assert my_flow.shape == (f-1, h, w, 2)

def test_show_flow(sample_tiff, title='Optical Flow', 
              step : int = 25, figsize : int | int = (12,6), scale : int = 200, 
              pivot : str = 'tail', color : str = 'blue', save_path : str = None):
    """
        Tests the show_flow function.
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)

    
    # Example preprocessing: normalize frames to 0-1, apply small Gaussian blur
    process_args = {
        "normalize": True,
        "gaussian_blur": 3
    }

    #arguments for testing
    flow_args = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0
    }

    my_flow = flow.calculate_optical_flow(img.arr, process_args, flow_args, False)
    video = flow.show_flow(my_flow, "Optical Flow", 25, (12,6), 200, 'tail', 'blue', None)
    assert isinstance(video, Figure)

    ax = video.axes[0]
    has_quiver = any(isinstance(c, Quiver) for c in ax.collections)
    assert has_quiver

    assert ax.get_title() == "Optical Flow"

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    assert xlim[0] == 0
    assert ylim[0] >= 0

    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"

def test_save_optflow_video(sample_tiff, tmp_path):
    """
       Tests the function that saves a video visualizing the optical flow.
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)
     # Example preprocessing: normalize frames to 0-1, apply small Gaussian blur
    process_args = {
        "normalize": True,
        "gaussian_blur": 3
    }

    #arguments for testing
    flow_args = {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": 0
        }
    save_path = tmp_path / "optflow_video.mp4"
    flow_calculated = flow.calculate_optical_flow(process_args, flow_args, False)
    my_video = flow.save_optflow_video(flow_calculated, save_path, 0, 20, 500, 
                                        'blue', 10, (12, 8), "Optical Flow Test", None, False)
    
    assert save_path.exists()
    assert save_path.suffix == ".mp4"       
    

def test_create_vector_field_video(sample_tiff):
    """
    Tests whether the create_vector_field_video function works correctly.
    """
    raise NotImplementedError

