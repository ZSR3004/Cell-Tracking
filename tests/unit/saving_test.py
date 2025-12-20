import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2, json, pytest, xyz_py, tifffile, matplotlib
from src.cell_tracking import tiffclass as tiff
from src.cell_tracking import saving as save
import matplotlib.pyplot as plt
from unittest.mock import patch, Mock, MagicMock, call
import numpy as np
from pathlib import Path
from scipy.io import savemat
import matplotlib.animation as animation
from src.cell_tracking.defaults import default_process, default_flow
from tests.unit.sample_tiffs import TIFF_PATHS


@pytest.fixture(params=TIFF_PATHS)
def init_tiff(request: pytest.FixtureRequest):
    """
    Creates a Tiff class instance.

    Args:
        request (pytest.FixtureRequest): The paths to generate Tiff instances from.

    Returns:
        (tiff.Tiff): A tuple containing information about the TIFF file.
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
    """
    path = request.param
    img = tifffile.imread(path)
    info = (img.shape[0], img.shape[1], img.shape[2], img.shape[3])
    return (tiff.Tiff(path), info)


def get_last_saved_pattern_fn_path(name: str, pattern_fn, main_path: str) -> Path:
    """
    An edited version of save.get_unique_path that gets the path with pattern_fn that was last saved.

    Args:
        name (str): Main identifier (e.g., protein name).
        pattern_fn (callable): Function that takes an integer and returns a file name.
        main_path (str): Main path to the directory.

    Returns:
        Path: Unique file path that does not yet exist.
    """
    save_dir = main_path
    save_dir.mkdir(parents=True, exist_ok=True)

    i = 1
    while True:
        file_name1 = pattern_fn(i)
        file_path1 = save_dir / file_name1
        if not file_path1.exists():
            file_name = pattern_fn(i - 1)
            file_path = save_dir / file_name
            return file_path
        i += 1


def test_get_unique_path(init_tiff: tuple, tmp_path):
    """
    Tests whether the get_unique_path method works correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file:
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
        tmp_path (pathlib.Path): A path to a temporary directory (this is a fixture in Pytest).

    Return:
        None
    """
    img, info = init_tiff
    f, c, h, w = info

    name1 = "flow"
    name2 = "Test_Name"
    name3 = "1Name"

    npy_fn_1 = lambda i: f"{name1}_flow{i}.npy"
    npy_fn_2 = lambda i: f"{name2}_flow{i}.npy"
    npy_fn_3 = lambda i: f"{name3}_flow{i}.npy"
    xyz_fn_1 = lambda i: f"{name1}_flow{i}.xyz"
    xyz_fn_2 = lambda i: f"{name2}_flow{i}.xyz"
    xyz_fn_3 = lambda i: f"{name3}_flow{i}.xyz"
    mat_fn_1 = lambda i: f"{name1}_flow{i}.mat"
    mat_fn_2 = lambda i: f"{name2}_flow{i}.mat"
    mat_fn_3 = lambda i: f"{name3}_flow{i}.mat"

    save_dir1 = tmp_path / name1
    save_dir2 = tmp_path / name2
    save_dir3 = tmp_path / name3

    assert not save_dir1.exists()
    assert not save_dir2.exists()
    assert not save_dir3.exists()

    unique_path_npy_fn_1 = save.get_unique_path(name1, npy_fn_1, save_dir1)
    assert save_dir1.exists()
    assert unique_path_npy_fn_1.name == "flow_flow1.npy"
    assert unique_path_npy_fn_1.parent == save_dir1

    unique_path_xyz_fn_2 = save.get_unique_path(name2, xyz_fn_2, save_dir2)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert unique_path_xyz_fn_2.name == "Test_Name_flow1.xyz"
    assert unique_path_xyz_fn_2.parent == save_dir2

    unique_path_mat_fn_3 = save.get_unique_path(name3, mat_fn_3, save_dir3)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_mat_fn_3.name == "1Name_flow1.mat"
    assert unique_path_mat_fn_3.parent == save_dir3

    (save_dir1 / "randomfile1.npy").touch()
    (save_dir1 / "flow_flow1.xyz").touch()
    (save_dir2 / "Test_Name_flow1.mat").touch()
    (save_dir2 / "randomfile2.xyz").touch()
    (save_dir3 / "randomfile1.mat").touch()
    (save_dir3 / "Test_Name_flow1.npy").touch()

    unique_path_npy_fn_2 = save.get_unique_path(name2, npy_fn_2, save_dir2)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_npy_fn_2.name == "Test_Name_flow1.npy"
    assert unique_path_npy_fn_2.parent == save_dir2

    unique_path_xyz_fn_3 = save.get_unique_path(name3, xyz_fn_3, save_dir3)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_xyz_fn_3.name == "1Name_flow1.xyz"
    assert unique_path_xyz_fn_3.parent == save_dir3

    unique_path_mat_fn_1 = save.get_unique_path(name1, mat_fn_1, save_dir1)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_mat_fn_1.name == "flow_flow1.mat"
    assert unique_path_mat_fn_1.parent == save_dir1

    (save_dir1 / "flow_flow1.xyz").touch()
    (save_dir1 / "flow_flow2.xyz").touch()
    (save_dir1 / "flow_flow3.xyz").touch()
    (save_dir1 / "flow_flow4.xyz").touch()
    (save_dir2 / "Test_Name_flow1.mat").touch()
    (save_dir2 / "Test_Name_flow2.mat").touch()
    (save_dir2 / "Test_Name_flow3.mat").touch()
    (save_dir2 / "Test_Name_flow4.mat").touch()
    (save_dir2 / "Test_Name_flow5.mat").touch()
    (save_dir2 / "Test_Name_flow6.mat").touch()
    (save_dir2 / "Test_Name_flow7.mat").touch()
    (save_dir2 / "Test_Name_flow8.mat").touch()
    (save_dir3 / "1Name_flow1.npy").touch()

    unique_path_npy_fn_3 = save.get_unique_path(name3, npy_fn_3, save_dir3)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_npy_fn_3.name == "1Name_flow2.npy"
    assert unique_path_npy_fn_3.parent == save_dir3

    unique_path_xyz_fn_1 = save.get_unique_path(name1, xyz_fn_1, save_dir1)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_xyz_fn_1.name == "flow_flow5.xyz"
    assert unique_path_xyz_fn_1.parent == save_dir1

    unique_path_mat_fn_2 = save.get_unique_path(name2, mat_fn_2, save_dir2)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_mat_fn_2.name == "Test_Name_flow9.mat"
    assert unique_path_mat_fn_2.parent == save_dir2


def test_save_arr(init_tiff: tuple, tmp_path):
    """
    Tests whether the save_arr method works correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file:
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
        tmp_path (pathlib.Path): A path to a temporary directory (this is a fixture in Pytest).

    Return:
        None
    """
    img, info = init_tiff
    f, c, h, w = info

    name1 = "Test_Name"
    save_dir = tmp_path / name1

    with (
        patch("src.cell_tracking.saving.get_unique_path") as mock_get_unique_path,
        patch("numpy.save") as mock_np_save
    ):
        save.save_arr(name1, img, save_dir)
        
        tiff_arr = img.arr
        save_arr1_path = get_last_saved_pattern_fn_path(
            name1, lambda i: f"{name1}_flow{i}.npy", save_dir
        )

        assert save_dir.exists()
        assert save_arr1_path.exists()

        assert np.array_equal(np.load(save_arr1_path), tiff_arr)


    #EDIT THIS: MOCK NP.SAVE, MOCK get_unique_path


def test_save_optical_flow_as_xyz(init_tiff: tuple, tmp_path):
    """
    Tests whether the save_optical_flow_as_xyz method works correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file:
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
        tmp_path (pathlib.Path): A path to a temporary directory (this is a fixture in Pytest).

    Return:
        None
    """
    img, info = init_tiff
    f, c, h, w = info
    tiff_arr = img.arr

    unshaped_arr1 = np.array(
        [[[[[1, 2]], [[3, 4]], [[5, 6]]]], [[[[7, 8]], [[9, 10]], [[11, 12]]]]]
    )
    shaped_arr1 = unshaped_arr1.reshape(-1, 2)
    assert np.array_equal(
        shaped_arr1, np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    )
    unshaped_arr2 = np.array(
        [
            [[[[4.2, 3.8]], [[9.5, 11.9]], [[0.1, 1.9]]]],
            [[[[7.0, 18.3829]], [[9.0, 1029.8]], [[4.3, 5.53]]]],
        ]
    )
    shaped_arr2 = unshaped_arr2.reshape(-1, 2)
    assert np.allclose(
        shaped_arr2,
        np.array(
            [
                [4.2, 3.8],
                [9.5, 11.9],
                [0.1, 1.9],
                [7.0, 18.3829],
                [9.0, 1029.8],
                [4.3, 5.53],
            ]
        ),
    )

    name1 = "Test_Name"
    save_dir = tmp_path / "save_dir"

    with patch("xyz_py.save_xyz") as mock_save_xyz:
        dx_dy_arr = tiff_arr.reshape(-1, 2)
        zeros = np.zeros((len(dx_dy_arr), 1), dtype=int)
        tiff_arr_xyz = np.hstack((dx_dy_arr, zeros))
        number_labels_arr = list(range(len(dx_dy_arr)))

        save.save_optical_flow_as_xyz(name1, tiff_arr, save_dir)
        save_path = save.get_unique_path(
            name1, lambda i: f"{name1}_flow{i}.xyz", save_dir
        )

        _, save_xyz_kwargs = mock_save_xyz.call_args
        assert save_xyz_kwargs["f_name"] == save_path
        assert np.array_equal(save_xyz_kwargs["labels"], number_labels_arr)
        assert np.array_equal(save_xyz_kwargs["coords"], tiff_arr_xyz)
        assert save_xyz_kwargs["coords"].shape == tiff_arr_xyz.shape
        assert len(save_xyz_kwargs["labels"]) == save_xyz_kwargs["coords"].shape[0]
        assert save_xyz_kwargs["comment"] == "Atoms"

        mock_save_xyz.assert_called_once()

    #EDIT THIS: MOCK xyz_py.save_xyz, MOCK get_unique_path


def test_save_optical_flow_as_matlab(init_tiff: tuple, tmp_path):
    """
    Tests whether the save_optical_flow_as_matlab method works correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file:
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
        tmp_path (pathlib.Path): A path to a temporary directory (this is a fixture in Pytest).

    Return:
        None
    """
    img, info = init_tiff
    f, c, h, w = info
    tiff_arr = img.arr

    name = "Test_Name"
    save_dir = tmp_path / "save_dir"

    with patch("src.cell_tracking.saving.savemat") as mock_savemat:
        save.save_optical_flow_as_matlab(name, tiff_arr, save_dir)

        args, kwargs = mock_savemat.call_args
        save_path = args[0]
        opt_flow_fortran = args[1]["optical_flow"]
        do_compression = kwargs["do_compression"]

        assert name in str(save_path)
        assert str(save_path).endswith(".mat")
        tiff_arr_fortran = np.asfortranarray(tiff_arr)
        assert np.array_equal(opt_flow_fortran, tiff_arr_fortran)
        assert opt_flow_fortran.shape == tiff_arr_fortran.shape
        assert do_compression == False
        mock_savemat.assert_called_once()

    #EDIT THIS: MOCK savemat, MOCK get_unique_path, MOCK np.asfortranarray ?


def test_save_optical_flow_as_numpy(init_tiff: tuple, tmp_path):
    """
    Tests whether the save_optical_flow_as_numpy method works correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file:
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
        tmp_path (pathlib.Path): A path to a temporary directory (this is a fixture in Pytest).

    Return:
        None
    """
    img, info = init_tiff
    f, c, h, w = info
    tiff_arr = img.arr

    name = "Test_Name"
    save_dir = tmp_path / "save_dir"

    with patch("numpy.save") as mock_save:
        save.save_optical_flow_as_numpy(name, tiff_arr, save_dir)

        args, _ = mock_save.call_args
        save_path = args[0]
        opt_flow = args[1]

        assert name in str(save_path)
        assert str(save_path).endswith(".npy")
        assert np.array_equal(opt_flow, tiff_arr)
        assert opt_flow.shape == tiff_arr.shape
        mock_save.assert_called_once()

    #EDIT THIS: MOCK NP.SAVE, MOCK get_unique_path


def test_save_original_video(init_tiff: tuple, tmp_path):
    """
    Tests whether the save_original_video method works correctly.

    Args:
        init_tiff (tuple): A tuple containing information about the TIFF file:
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
        tmp_path (pathlib.Path): A path to a temporary directory (this is a fixture in Pytest).

    Return:
        None
    """
    img, info = init_tiff
    f, c, h, w = info
    tiff_arr = img.arr

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_im = MagicMock()

    mock_im.set_data = MagicMock()
    mock_ax.set_title = MagicMock()

    kwargs1 = {"T": 10, "fps": 20}
    kwargs2 = {"T": 40}
    kwargs3 = {"fps": 35}
    kwargs4 = {}

    image_stack1 = np.asarray(tiff_arr[:, 0, :, :])
    image_stack2 = np.asarray(tiff_arr[:, 1, :, :])
    image_stack3 = np.asarray(tiff_arr[:, 2, :, :])

    stack1_kwargs1_path = tmp_path / "stack1_kwargs1.mp4"
    stack1_kwargs2_path = tmp_path / "stack1_kwargs2.mp4"
    stack1_kwargs3_path = tmp_path / "stack1_kwargs3.mp4"
    stack1_kwargs4_path = tmp_path / "stack1_kwargs4.mp4"
    stack2_kwargs1_path = tmp_path / "stack2_kwargs1.mp4"
    stack2_kwargs2_path = tmp_path / "stack2_kwargs2.mp4"
    stack2_kwargs3_path = tmp_path / "stack2_kwargs3.mp4"
    stack2_kwargs4_path = tmp_path / "stack2_kwargs4.mp4"
    stack3_kwargs1_path = tmp_path / "stack3_kwargs1.mp4"
    stack3_kwargs2_path = tmp_path / "stack3_kwargs2.mp4"
    stack3_kwargs3_path = tmp_path / "stack3_kwargs3.mp4"
    stack3_kwargs4_path = tmp_path / "stack3_kwargs4.mp4"

    def test_case_x(image_stackx: np.ndarray, stackx_kwargsx_path: str, kwargsx: dict):
        """
        Tests whether the save_original_video method works correctly on a given test case.

        Args:
            image_stackx (np.ndarray): Image stack of shape (T, H, W).
            stackx_kwargsx_path (str): The path to save the video file to.
            kwargsx (dict): Additional keyword arguments that include:
                - T (int): Total number of frames in the image stack.
                - fps (int): Frames per second for the video.

        Returns:
            None.
        """
        with (
            patch(
                "src.cell_tracking.saving.animation.FFMpegWriter"
            ) as mock_ffmpegwriter,
            patch(
                "src.cell_tracking.saving.animation.FuncAnimation"
            ) as mock_funcanimation,
        ):
            mock_ani = MagicMock()
            mock_funcanimation.return_value = mock_ani
            mock_writer_instance = MagicMock()
            mock_ffmpegwriter.return_value = mock_writer_instance

            save.save_original_video(
                "Video_Name",
                stackx_kwargsx_path,
                mock_im,
                image_stackx,
                mock_fig,
                mock_ax,
                **kwargsx,
            )

            T = kwargsx.get("T", image_stackx.shape[0])
            fps = kwargsx.get("fps", 10)

            args_FuncAnimation, kwargs_FuncAnimation = mock_funcanimation.call_args
            assert args_FuncAnimation[0] == mock_fig
            assert callable(args_FuncAnimation[1])
            assert kwargs_FuncAnimation["frames"] == T
            assert kwargs_FuncAnimation["interval"] == 1000 / fps
            assert kwargs_FuncAnimation["blit"] == False

            _, kwargs_FFMpegWriter = mock_ffmpegwriter.call_args
            assert kwargs_FFMpegWriter["fps"] == fps

            mock_ani.save.assert_called_once_with(
                stackx_kwargsx_path, writer=mock_writer_instance
            )
            mock_funcanimation.assert_called_once()
            mock_ffmpegwriter.assert_called_once()

    test_case_x(image_stack1, stack1_kwargs1_path, kwargs1)
    test_case_x(image_stack1, stack1_kwargs2_path, kwargs2)
    test_case_x(image_stack1, stack1_kwargs3_path, kwargs3)
    test_case_x(image_stack1, stack1_kwargs4_path, kwargs4)
    test_case_x(image_stack2, stack2_kwargs1_path, kwargs1)
    test_case_x(image_stack2, stack2_kwargs2_path, kwargs2)
    test_case_x(image_stack2, stack2_kwargs3_path, kwargs3)
    test_case_x(image_stack2, stack2_kwargs4_path, kwargs4)
    test_case_x(image_stack3, stack3_kwargs1_path, kwargs1)
    test_case_x(image_stack3, stack3_kwargs2_path, kwargs2)
    test_case_x(image_stack3, stack3_kwargs3_path, kwargs3)
    test_case_x(image_stack3, stack3_kwargs4_path, kwargs4)
