import sys, os, pytest
from src import tiffclass as tiff
import numpy as np

def test_init(path: str):
    """
    Tests whether the Tiff class initializes correctly.
    """
    img = tiff.Tiff(path=path, n_channels=3, dtype=np.uint16)

    assert img.path == str(path)
    assert img.n_channels == 3
    assert img.dtype == np.uint16
    assert isinstance(img.arr, np.ndarray)
    assert hasattr(img, "timestamp")
    assert hasattr(img, "arr")
    assert hasattr(img, "tags")

    assert img.arr.shape[0] == 96
    assert img.arr.shape[1] == 3
    assert img.arr.shape[2] == 520
    assert img.arr.shape[3] == 2329


def test_isolate_channel(path: str):
    """
    Tests whether the isolate_channel method works correctly.
    """
    img = tiff.Tiff(path=path, n_channels=3, dtype=np.uint16)

    channel_0 = img.isolate_channel(0)
    channel_1 = img.isolate_channel(1)
    channel_2 = img.isolate_channel(2)

    assert isinstance(channel_0, np.ndarray)
    assert isinstance(channel_1, np.ndarray)
    assert isinstance(channel_2, np.ndarray)

    assert channel_0.shape == (96, 520, 2329)
    assert channel_1.shape == (96, 520, 2329)
    assert channel_2.shape == (96, 520, 2329)

    assert np.array_equal(channel_0, img.arr[:, 0, :, :])
    assert np.array_equal(channel_1, img.arr[:, 1, :, :])
    assert np.array_equal(channel_2, img.arr[:, 2, :, :])

    assert not np.array_equal(channel_0, channel_1)
    assert not np.array_equal(channel_1, channel_2)
    assert not np.array_equal(channel_0, channel_2)


def run_tiffclass_test_suite(path_list: list[str]):
    for path in path_list:
        test_init(path)
        test_isolate_channel(path)


if __name__ == "__main__":
    path_list = [
        "../../datasets/nuclei_labeled/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif"
    ]

    run_tiffclass_test_suite(path_list)
