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


