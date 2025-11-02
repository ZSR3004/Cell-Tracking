import sys, os, pytest
from src import optical_flow as flow
import numpy as np
from src import tiffclass as tiff

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
    raise NotImplementedError

def test_optical_flow(sample_tiff):
    """
    Tests whether the optical_flow function works correctly.
    """
    raise NotImplementedError

def test_calculate_optical_flow(self, process_args=None, flow_args=None, default=False):
        """
        Tests whether the calculate_optical_flow function works correctly.
        """
        raise NotImplementedError

def test_save_optflow_video(flow, idx : int = 0, step : int = 20, 
                          scale : int = 500, color : str = 'blue', fps : int = 10, 
                          figsize : int | int = (12,8),
                          title : str = None, overlay : bool = False):
     """
        Saves a video visualizing the optical flow.
     """
     raise NotImplementedError

def test_create_vector_field_video(name, arr : np.ndarray, og_arr : np.ndarray=None, 
                    step : int = 20, scale : int = 500, color : str = 'blue', 
                    fps : int = 10, figsize : int | int = (12,8),
                    title : str = None, flag : str = None) -> None:
    """
    Tests whether the create_vector_field_video function works correctly.
    """
    raise NotImplementedError

def test_calculate_trajectory(flow):
    """
    Tests whether the calculate_trajectory function works correctly.
    """
    raise NotImplementedError

