import sys, os, pytest
from src import tiffclass as tiff
import numpy as np

@pytest.fixture
def sample_tiff():
    path = "../../datasets/nuclei_labeled/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif"
    f, h, w = 96, 520, 2329
    return tiff.Tiff(path), f, h, w

def test_init(sample_tiff):
    """
    Tests whether the Tiff class initializes correctly.
    """
    path, f, h, w = sample_tiff
    img = tiff.Tiff(path)

    assert img.path == str(path)
    assert img.n_channels == 3
    assert img.dtype == np.uint16
    assert isinstance(img.arr, np.ndarray)
    assert hasattr(img, "timestamp")
    assert hasattr(img, "arr")
    assert hasattr(img, "tags")

    assert img.arr.shape[0] == f  # number of frames
    assert img.arr.shape[1] == 3  # number of channels
    assert img.arr.shape[2] == h  # height
    assert img.arr.shape[3] == w  # width


def test_isolate_channel(sample_tiff):
    """
    Tests whether the isolate_channel method works correctly.
    """
    path, f, h, w = sample_tiff
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


def run_tiffclass_test_suite(path_list: list[dict]):
    """
    Runs all tests on a list of paths.
    """
    for path in path_list:
        test_init(path["path"], path["frames"], path["height"], path["width"])
        test_isolate_channel(
            path["path"], path["frames"], path["height"], path["width"]
        )


def make_dict_of_path(path: str, frames: int, height: int, width: int) -> dict:
    """
    Makes a dictionary out of a path to a tiff and its metadata.
    """
    return {"path": path, "frames": frames, "height": height, "width": width}


if __name__ == "__main__":
    path_list = [
        make_dict_of_path(
            "../../datasets/nuclei_labeled/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif",
            96,
            520,
            2329,
        )
    ]

    run_tiffclass_test_suite(path_list)
