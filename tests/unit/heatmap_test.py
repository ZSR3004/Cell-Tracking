import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2, pytest
from src.cell_tracking import heatmap
from src.cell_tracking import tiffclass as tiff
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from unittest.mock import patch, Mock, MagicMock

TIFF_PATHS = [
    (
        "datasets/nuclei_labeled/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif",
        (96, 3, 520, 2329),
    )
]


@pytest.fixture(params=TIFF_PATHS)
def init_tiff(request: pytest.FixtureRequest) -> tiff.Tiff:
    """
    Creates a Tiff class instance.

    Args:
        request (pytest.FixtureRequest): The paths to generate Tiff instances from.

    Returns:
        (tiff.Tiff): Tiff class instance of the path.
    """
    path, info = request.param
    return (tiff.Tiff(path), info)


def test_vector_magnitude_heatmaps(init_tiff):
    """
    Tests whether the vector_magnitude_heatmaps function works correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file.
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.

    Returns:
        None.
    """
    img, info = init_tiff
    f, c, h, w = info
    tiff_arr = img.arr

    def dummy_optical_flow(channel: int):
        """
        A function that creates an array similar to (and with the same shape as) flow.optical_flow(tiff_arr, channel). This function is much
        faster than calling flow.optical_flow (which is why it's perfect for testing).

        Args:
            channel (int): The channel to process.

        Returns:
            A np.ndarray of shape (f-1, h, w, 2).
        """
        arr_channel = tiff_arr[:, channel, :, :]

        dummy_dx = arr_channel[1:] - arr_channel[:-1]
        dummy_dy = arr_channel[1:] - arr_channel[:-1]

        return np.stack([dummy_dx, dummy_dy], axis=-1)
    
    #flow0, flow1, and flow2 have shape (f-1, h, w, 2), which is the same shape as flow.optical_flow(tiff_arr, n) (where n is 0, 1, or 2)
    flow0 = dummy_optical_flow(0)
    flow1 = dummy_optical_flow(1)
    flow2 = dummy_optical_flow(2)

    def test_case_x(flowx: np.ndarray, normalizex: bool):
        """
        Tests whether the vector_magnitude_heatmaps function works correctly on a specific test case.

        Args:
            flowx (np.ndarray): Flow array of shape (frames, height, width, 2).
            normalizex (bool): If True, normalizes magnitudes to 0-255 range for visualization.

        Returns:
            None.
        """
        with patch("numpy.linalg.norm") as mock_linalg_norm, \
            patch("cv2.normalize") as mock_cv2_normalize:
            result = heatmap.vector_magnitude_heatmaps(flowx, normalizex)

            #fake_flow has shape (f, h, w), which is the same shape as np.linalg.norm(flowx, axis=-1)
            fake_flow = flowx[..., 0]
            mock_linalg_norm.return_value = fake_flow

            if normalizex:
                #assert mock_cv2_normalize is called the right number of times
                #do more

            else: