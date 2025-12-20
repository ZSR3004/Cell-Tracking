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
from tests.unit.sample_tiffs import TIFF_PATHS
from src.cell_tracking.defaults import default_process, default_flow


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
    tiff_arr = img.arr

    name1 = "Test_Name"
    save_dir = tmp_path / name1
    assert not save_dir.exists()

    save.save_arr(name1, img, save_dir)
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
            mock_im.set_data = MagicMock()
            mock_ax.set_title = MagicMock()
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


def test_save_vector_video(init_tiff: tuple, tmp_path):
    """
    Tests whether the save_vector_video method works correctly.

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

    def dummy_arr(channel: int):
        """
        A function that creates an array of shape (f-1, h, w, 2).

        Args:
            channel (int): The channel to process.

        Returns:
            A np.ndarray of shape (f-1, h, w, 2).
        """
        arr_channel = tiff_arr[:, channel, :, :]

        dummy_dx = arr_channel[1:] - arr_channel[:-1]
        dummy_dy = arr_channel[1:] - arr_channel[:-1]

        return np.stack([dummy_dx, dummy_dy], axis=-1)

    # arr0, arr1, and arr2 have shape (f-1, h, w, 2)
    arr0 = dummy_arr(0)
    arr1 = dummy_arr(1)
    arr2 = dummy_arr(2)

    mock_img_disp = MagicMock()
    mock_quiver = MagicMock()
    mock_ax = MagicMock()
    mock_fig = MagicMock()

    name1 = "save_vector_video0"
    name2 = "Test Save"

    flag0 = ''
    flag00 = ' '
    flag000 = 'ft'
    flag0000 = 'f '
    flag1 = 'f'
    flag2 = 't'

    og_arr0 = None
    og_arr1 = tiff_arr

    step1 = 10
    step2 = 35
    step3 = 20

    fps1 = 5
    fps2 = 32
    fps3 = 10

    T = f-1

    file_path1 = tmp_path / "path1"
    file_path2 = tmp_path / "path2"
    file_path3 = tmp_path / "path3"
    file_path4 = tmp_path / "path4"
    file_path5 = tmp_path / "path5"
    file_path6 = tmp_path / "path6"
    file_path7 = tmp_path / "path7"
    file_path8 = tmp_path / "path8"
    file_path9 = tmp_path / "path9"
    file_path10 = tmp_path / "path10"
    file_path11 = tmp_path / "path11"
    file_path12 = tmp_path / "path12"
    file_path13 = tmp_path / "path13"
    file_path14 = tmp_path / "path14"
    file_path15 = tmp_path / "path15"
    file_path16 = tmp_path / "path16"
    file_path17 = tmp_path / "path17"
    file_path18 = tmp_path / "path18"
    file_path19 = tmp_path / "path19"
    file_path20 = tmp_path / "path20"

    kwargs1 = {'img_disp': mock_img_disp, 'arr': arr0, 'og_arr': og_arr1, 'step': step1, 'fps': fps1, 'quiver': mock_quiver, 'ax': mock_ax, 'fig': mock_fig, 'T': T}
    kwargs2 = {'img_disp': mock_img_disp, 'arr': arr1, 'og_arr': og_arr1, 'step': step2, 'fps': fps2, 'quiver': mock_quiver, 'ax': mock_ax, 'fig': mock_fig, 'T': T}
    kwargs3 = {'img_disp': mock_img_disp, 'arr': arr2, 'og_arr': og_arr1, 'step': step3, 'fps': fps3, 'quiver': mock_quiver, 'ax': mock_ax, 'fig': mock_fig, 'T': T}
    kwargs4 = {'arr': arr0, 'step': step1, 'fps': fps3, 'quiver': mock_quiver, 'ax': mock_ax, 'fig': mock_fig, 'T': T}
    kwargs5 = {'img_disp': mock_img_disp, 'arr': arr1, 'og_arr': og_arr0, 'fps': fps1, 'quiver': mock_quiver, 'ax': mock_ax, 'fig': mock_fig, 'T': T}
    kwargs6 = {'img_disp': mock_img_disp, 'arr': arr2, 'fps': fps1, 'quiver': mock_quiver, 'ax': mock_ax, 'fig': mock_fig, 'T': T}
    kwargs7 = {'img_disp': mock_img_disp, 'arr': arr0, 'og_arr': og_arr1, 'quiver': mock_quiver, 'ax': mock_ax, 'fig': mock_fig, 'T': T}
    kwargs8 = {'arr': arr1, 'og_arr': og_arr1, 'step': step2, 'fps': fps2, 'quiver': mock_quiver, 'ax': mock_ax, 'fig': mock_fig, 'T': T}
    kwargs9 = {'arr': arr2, 'og_arr': og_arr0, 'step': step3, 'quiver': mock_quiver, 'ax': mock_ax, 'fig': mock_fig, 'T': T}
    kwargs10 = {'arr': arr1, 'quiver': mock_quiver, 'ax': mock_ax, 'fig': mock_fig, 'T': T}

    def test_case_x(namex: str, flagx: str, file_pathx: str, **kwargsx: dict):
        """
        Tests whether the save_vector_video method works correctly on a given test case.

        Args:
            name (str): Name of the video file to save.
            flag (str): Flag to determine the type of video being saved.
            file_pathx (str): Path that the vector video will be saved to. Return value of mocked get_unique_path.
            **kwargs: Additional keyword arguments that include:
                - img_disp: Matplotlib image display object for the original frames.
                - arr: Optical flow array of shape (f-1, h, w, 2) where f-1 is the number of frames,
                    h is height, w is width, and the last dimension contains the flow vectors (dx, dy).
                - og_arr: Original image frames array of shape (f, c, h, w). Default is None.
                - step: Step size for downsampling the flow vectors for visualization. Default is 20.
                - fps: Frames per second for the video. Default is 10.
                - quiver: Matplotlib quiver object for displaying flow vectors.
                - ax: Matplotlib axes object for the plot.
                - fig: Matplotlib figure object for the plot.
                - T: Total number of frames minus one (T-1).

        Returns:
            None.
        """
        with (
            patch("src.cell_tracking.saving.get_unique_path") as mock_get_unique_path,
            patch("src.cell_tracking.saving.animation.FuncAnimation") as mock_funcanimation,
            patch("src.cell_tracking.saving.animation.writers") as mock_ani_writers
        ):
            mock_quiver.set_UVC = MagicMock()
            mock_img_disp.set_data = MagicMock()
            mock_ax.set_title = MagicMock()

            mock_get_unique_path.return_value = file_pathx
            mock_ani = MagicMock()
            mock_funcanimation.return_value = mock_ani
            mock_Writer = MagicMock()
            mock_ani_writers.__getitem__.return_value = mock_Writer
            mock_writer = MagicMock()
            mock_Writer.return_value = mock_writer

            img_disp = kwargsx.get('img_disp', None)
            arr = kwargsx['arr']
            og_arr = kwargsx.get('og_arr', None)
            step = kwargsx.get('step', 20)
            fpsx = kwargsx.get('fps', 10)
            quiver = kwargsx['quiver']
            ax = kwargsx['ax']
            figx = kwargsx['fig']
            Tx = kwargsx['T']

            if flagx not in ['f', 't']:
                match_message = f'Invalid flag. Expected f or t, but got {flagx}'
                with pytest.raises(ValueError, match=match_message):
                    save.save_vector_video(namex, flagx, **kwargsx)
            else:
                save.save_vector_video(namex, flagx, **kwargsx)
                
                args_get_unique_path, _ = mock_get_unique_path.call_args
                assert args_get_unique_path[0] == namex
                assert args_get_unique_path[1] == 'video'
                assert callable(args_get_unique_path[2])

                args_mock_funcanimation, kwargs_mock_funcanimation = mock_funcanimation.call_args
                assert args_mock_funcanimation[0] == figx
                assert callable(args_mock_funcanimation[1])
                assert kwargs_mock_funcanimation['frames'] == range(Tx)
                assert kwargs_mock_funcanimation['interval'] == 1000/fpsx
                assert kwargs_mock_funcanimation['blit'] == False

                _, kwargs_Writer = mock_Writer.call_args
                assert kwargs_Writer['fps'] == fpsx
                assert kwargs_Writer['metadata'] == dict(artist='Flow')
                assert kwargs_Writer['bitrate'] == 1800

                args_ani_save, kwargs_ani_save = mock_ani.save.call_args
                assert args_ani_save[0] == file_pathx
                assert kwargs_ani_save['writer'] == mock_writer

                mock_get_unique_path.assert_called_once()
                mock_funcanimation.assert_called_once()
                mock_ani_writers.__getitem__.assert_called_once_with('ffmpeg')
                mock_Writer.assert_called_once()
                mock_ani.save.assert_called_once()

    test_case_x(name1, flag0, file_path1, **kwargs1)
    test_case_x(name2, flag1, file_path2, **kwargs2)
    test_case_x(name1, flag2, file_path3, **kwargs3)
    test_case_x(name2, flag2, file_path4, **kwargs4)
    test_case_x(name1, flag1, file_path5, **kwargs5)
    test_case_x(name2, flag2, file_path6, **kwargs6)
    test_case_x(name1, flag00, file_path7, **kwargs7)
    test_case_x(name2, flag1, file_path8, **kwargs8)
    test_case_x(name1, flag1, file_path9, **kwargs9)
    test_case_x(name2, flag2, file_path10, **kwargs10)
    test_case_x(name1, flag2, file_path11, **kwargs1)
    test_case_x(name2, flag2, file_path12, **kwargs2)
    test_case_x(name1, flag000, file_path13, **kwargs3)
    test_case_x(name2, flag1, file_path14, **kwargs4)
    test_case_x(name1, flag2, file_path15, **kwargs5)
    test_case_x(name2, flag1, file_path16, **kwargs6)
    test_case_x(name1, flag1, file_path17, **kwargs7)
    test_case_x(name2, flag0000, file_path18, **kwargs8)
    test_case_x(name1, flag2, file_path19, **kwargs9)
    test_case_x(name2, flag1, file_path20, **kwargs10)