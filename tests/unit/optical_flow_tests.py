import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import sys, os, pytest
from sympy import Idx
from src import optical_flow as flow
import numpy as np
from src import tiffclass as tiff
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
import cv2

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

    flow_0 = flow.optical_flow(img.arr, 0)
    flow_1 = flow.optical_flow(img.arr, 1)
    flow_2 = flow.optical_flow(img.arr, 2)

    combine_flow_0 = flow.combine_flows([flow_0, flow_1])
    combine_flow_1 = flow.combine_flows([flow_1, flow_2])
    combine_flow_2 = flow.combine_flows([flow_0, flow_2])


    assert combine_flow_0.shape == (f-1, c, h, w, 2)
    assert combine_flow_1.shape == (f-1, c, h, w, 2)
    assert combine_flow_2.shape == (f-1, c, h, w, 2)


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

    my_flow = flow.optical_flow(img.arr, 0, **flow_args)

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
    "normalize": {"alpha": 0, "beta": 255, "norm_type": cv2.NORM_MINMAX},
    "gauss": {"ksize": (5, 5), "sigmaX": 1.5}}

    my_flow = flow.calculate_optical_flow(img.arr, process_args)

    # Test output type
    assert isinstance(my_flow, np.ndarray)

    # Test output shape
    assert my_flow.shape == (f-1, c, h, w, 2)

def test_show_flow(sample_tiff, title='Optical Flow', 
              step : int = 25, figsize : int | int = (12,6), scale : int = 200, 
              pivot : str = 'tail', color : str = 'blue', save_path : str = None):
    """
        Tests the show_flow function.
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)

    my_flow = flow.optical_flow(img.arr, 0)
    first_flow_frame = my_flow[0]
    video = flow.show_flow(first_flow_frame, "Optical Flow", 25, (12,6), 200, 'tail', 'blue', None)

    fig = plt.gcf()
    assert isinstance(fig, Figure)
    plt.close()
       
    

def test_create_vector_field_video(sample_tiff):
    """
    Tests whether the create_vector_field_video function works correctly.
    """
    path, f, c, h, w = sample_tiff
    img = tiff.Tiff(path)
    
    flow_calculated = flow.optical_flow(img.arr, 0)
    my_video = flow.create_vector_field_video("vector_video.mp4", flow_calculated)

    fig = plt.gcf()

    assert isinstance(fig, Figure)
    plt.close()  

