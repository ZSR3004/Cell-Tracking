import tiffclass as tiff
from tiffclass import np
import pytest
from tiffclass import tf

def test_init(tmp_path):
    """
    Tests whether the Tiff class initializes correctly.
    """
    fake_tiff_path = tmp_path / "test.tiff"
    fake_tiff_path.write_bytes(b"Fake TIFF data")

    x = tiff.Tiff(path=str(fake_tiff_path), n_channels=3, dtype=np.uint16)

    assert x.path == str(fake_tiff_path)
    assert x.n_channels == 3
    assert x.dtype == np.uint16
    assert hasattr(x, "timestamp")
    assert hasattr(x, "arr")
    assert hasattr(x, "tags")

    


