import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2, json, pytest, gc, shutil, xyz_py
from src.cell_tracking import tiffclass as tiff
from src.cell_tracking import saving as save
from src.cell_tracking import optical_flow as flow
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import matplotlib.animation as animation
from src.cell_tracking.defaults import default_process, default_flow

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
    save_dir = main_path / name
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

    unique_path_npy_fn_1 = save.get_unique_path(name1, npy_fn_1, tmp_path)
    assert save_dir1.exists()
    assert unique_path_npy_fn_1.name == "flow_flow1.npy"
    assert unique_path_npy_fn_1.parent == save_dir1

    unique_path_xyz_fn_2 = save.get_unique_path(name2, xyz_fn_2, tmp_path)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert unique_path_xyz_fn_2.name == "Test_Name_flow1.xyz"
    assert unique_path_xyz_fn_2.parent == save_dir2

    unique_path_mat_fn_3 = save.get_unique_path(name3, mat_fn_3, tmp_path)
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

    unique_path_npy_fn_2 = save.get_unique_path(name2, npy_fn_2, tmp_path)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_npy_fn_2.name == "Test_Name_flow1.npy"
    assert unique_path_npy_fn_2.parent == save_dir2

    unique_path_xyz_fn_3 = save.get_unique_path(name3, xyz_fn_3, tmp_path)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_xyz_fn_3.name == "1Name_flow1.xyz"
    assert unique_path_xyz_fn_3.parent == save_dir3

    unique_path_mat_fn_1 = save.get_unique_path(name1, mat_fn_1, tmp_path)
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

    unique_path_npy_fn_3 = save.get_unique_path(name3, npy_fn_3, tmp_path)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_npy_fn_3.name == "1Name_flow2.npy"
    assert unique_path_npy_fn_3.parent == save_dir3

    unique_path_xyz_fn_1 = save.get_unique_path(name1, xyz_fn_1, tmp_path)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_xyz_fn_1.name == "flow_flow5.xyz"
    assert unique_path_xyz_fn_1.parent == save_dir1

    unique_path_mat_fn_2 = save.get_unique_path(name2, mat_fn_2, tmp_path)
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

    save_arr1 = save.save_arr(name1, img, tmp_path)
    save_arr1_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.npy", tmp_path
    )

    assert save_dir.exists()
    assert save_arr1_path.exists()

    assert np.array_equal(np.load(save_arr1_path), tiff_arr)


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

    unshaped_arr = np.array(
        [[[[[1, 2]], [[3, 4]], [[5, 6]]]], [[[[7, 8]], [[9, 10]], [[11, 12]]]]]
    )
    shaped_arr = unshaped_arr.reshape(-1, 2)
    assert np.array_equal(
        shaped_arr, np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    )

    name1 = "Test_Name"

    kwargs1 = {
        "pyr_scale": 0.25,
        "levels": 2,
        "winsize": 30,
        "iterations": 5,
        "poly_n": 7,
        "poly_sigma": 2.8,
        "flags": 1,
    }

    optical_flow_channel0 = flow.optical_flow(arr=tiff_arr, channel=0)
    dx_dy_arr1 = optical_flow_channel0.reshape(-1, 2)
    zeros1 = np.zeros((len(dx_dy_arr1), 1), dtype=dx_dy_arr1[0][0].dtype)
    optical_flow_channel0_xyz = np.hstack((dx_dy_arr1, zeros1))
    save_dir1 = tmp_path / "save_dir1"
    assert not save_dir1.exists()
    save.save_optical_flow_as_xyz(name1, optical_flow_channel0, save_dir1)
    optical_flow_channel0_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.xyz", save_dir1
    )
    assert save_dir1.exists()
    assert optical_flow_channel0_path.exists()
    optical_flow_channel0_arr = xyz_py.load_xyz(optical_flow_channel0_path)
    labels, coords = optical_flow_channel0_arr
    assert isinstance(labels, list)
    assert isinstance(coords, np.ndarray)
    assert np.array_equal(coords, optical_flow_channel0_xyz)
    assert coords.shape == optical_flow_channel0.shape
    assert labels.shape == len(optical_flow_channel0)
    del optical_flow_channel0, optical_flow_channel0_arr
    gc.collect()
    shutil.rmtree(save_dir1)

    optical_flow_channel1 = flow.optical_flow(arr=tiff_arr, channel=1)
    dx_dy_arr2 = optical_flow_channel1.reshape(-1, 2)
    zeros2 = np.zeros((len(dx_dy_arr2), 1), dtype=dx_dy_arr2[0][0].dtype)
    optical_flow_channel1_xyz = np.hstack((dx_dy_arr2, zeros2))
    save_dir2 = tmp_path / "save_dir2"
    assert not save_dir2.exists()
    save.save_optical_flow_as_xyz(name1, optical_flow_channel1, save_dir2)
    optical_flow_channel1_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.xyz", save_dir2
    )
    assert save_dir2.exists()
    assert optical_flow_channel1_path.exists()
    optical_flow_channel1_arr = xyz_py.load_xyz(optical_flow_channel1_path)
    labels, coords = optical_flow_channel1_arr
    assert isinstance(labels, list)
    assert isinstance(coords, np.ndarray)
    assert np.array_equal(optical_flow_channel1_arr, optical_flow_channel1_xyz)
    assert coords.shape == optical_flow_channel1.shape
    assert labels.shape == len(optical_flow_channel1)
    del optical_flow_channel1, optical_flow_channel1_arr
    gc.collect()
    shutil.rmtree(save_dir2)

    optical_flow_channel2 = flow.optical_flow(arr=tiff_arr, channel=2)
    dx_dy_arr3 = optical_flow_channel2.reshape(-1, 2)
    zeros3 = np.zeros((len(dx_dy_arr3), 1), dtype=dx_dy_arr3[0][0].dtype)
    optical_flow_channel2_xyz = np.hstack((dx_dy_arr3, zeros3))
    save_dir3 = tmp_path / "save_dir3"
    assert not save_dir3.exists()
    save.save_optical_flow_as_xyz(name1, optical_flow_channel2, save_dir3)
    optical_flow_channel2_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.xyz", save_dir3
    )
    assert save_dir3.exists()
    assert optical_flow_channel2_path.exists()
    optical_flow_channel2_arr = xyz_py.load_xyz(optical_flow_channel2_path)
    labels, coords = optical_flow_channel2_arr
    assert isinstance(labels, list)
    assert isinstance(coords, np.ndarray)
    assert np.array_equal(optical_flow_channel2_arr, optical_flow_channel2_xyz)
    assert coords.shape == optical_flow_channel2.shape
    assert labels.shape == len(optical_flow_channel2)
    del optical_flow_channel2, optical_flow_channel2_arr
    gc.collect()
    shutil.rmtree(save_dir3)

    optical_flow_channel0_custom = flow.optical_flow(arr=tiff_arr, channel=0, **kwargs1)
    dx_dy_arr4 = optical_flow_channel0_custom.reshape(-1, 2)
    zeros4 = np.zeros((len(dx_dy_arr4), 1), dtype=dx_dy_arr4[0][0].dtype)
    optical_flow_channel0_custom_xyz = np.hstack((dx_dy_arr4, zeros4))
    save_dir4 = tmp_path / "save_dir4"
    assert not save_dir4.exists()
    save.save_optical_flow_as_xyz(name1, optical_flow_channel0_custom, save_dir4)
    optical_flow_channel0_custom_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.xyz", save_dir4
    )
    assert save_dir4.exists()
    assert optical_flow_channel0_custom_path.exists()
    optical_flow_channel0_custom_arr = xyz_py.load_xyz(
        optical_flow_channel0_custom_path
    )
    labels, coords = optical_flow_channel0_custom_arr
    assert isinstance(labels, list)
    assert isinstance(coords, np.ndarray)
    assert np.array_equal(
        optical_flow_channel0_custom_arr, optical_flow_channel0_custom_xyz
    )
    assert coords.shape == optical_flow_channel0_custom.shape
    assert labels.shape == len(optical_flow_channel0_custom)
    del optical_flow_channel0_custom, optical_flow_channel0_custom_arr
    gc.collect()
    shutil.rmtree(save_dir4)

    calculate_optical_flow = flow.calculate_optical_flow(arr=tiff_arr)
    dx_dy_arr5 = calculate_optical_flow.reshape(-1, 2)
    zeros5 = np.zeros((len(dx_dy_arr5), 1), dtype=dx_dy_arr5[0][0].dtype)
    calculate_optical_flow_xyz = np.hstack((dx_dy_arr5, zeros5))
    save_dir5 = tmp_path / "save_dir5"
    assert not save_dir5.exists()
    save.save_optical_flow_as_xyz(name1, calculate_optical_flow, save_dir5)
    calculate_optical_flow_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.xyz", save_dir5
    )
    assert save_dir5.exists()
    assert calculate_optical_flow_path.exists()
    calculate_optical_flow_arr = xyz_py.load_xyz(calculate_optical_flow_path)
    labels, coords = calculate_optical_flow_arr
    assert isinstance(labels, list)
    assert isinstance(coords, np.ndarray)
    assert np.array_equal(calculate_optical_flow_arr, calculate_optical_flow_xyz)
    assert coords.shape == calculate_optical_flow.shape
    assert labels.shape == len(calculate_optical_flow)
    del calculate_optical_flow, calculate_optical_flow_arr
    gc.collect()
    shutil.rmtree(save_dir5)

    calculate_optical_flow_default_true = flow.calculate_optical_flow(arr=tiff_arr)
    dx_dy_arr6 = calculate_optical_flow_default_true.reshape(-1, 2)
    zeros6 = np.zeros((len(dx_dy_arr6), 1), dtype=dx_dy_arr6[0][0].dtype)
    calculate_optical_flow_default_true_xyz = np.hstack((dx_dy_arr6, zeros6))
    save_dir6 = tmp_path / "save_dir6"
    assert not save_dir6.exists()
    save.save_optical_flow_as_xyz(name1, calculate_optical_flow_default_true, save_dir6)
    calculate_optical_flow_default_true_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.xyz", save_dir6
    )
    assert save_dir6.exists()
    assert calculate_optical_flow_default_true_path.exists()
    calculate_optical_flow_default_true_arr = xyz_py.load_xyz(
        calculate_optical_flow_default_true_path
    )
    labels, coords = calculate_optical_flow_default_true_arr
    assert isinstance(labels, list)
    assert isinstance(coords, np.ndarray)
    assert np.array_equal(
        calculate_optical_flow_default_true_arr, calculate_optical_flow_default_true_xyz
    )
    assert coords.shape == calculate_optical_flow_default_true.shape
    assert labels.shape == len(calculate_optical_flow_default_true)
    del calculate_optical_flow_default_true, calculate_optical_flow_default_true_arr
    gc.collect()
    shutil.rmtree(save_dir6)


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

    name1 = "Test_Name"

    kwargs1 = {
        "pyr_scale": 0.25,
        "levels": 2,
        "winsize": 30,
        "iterations": 5,
        "poly_n": 7,
        "poly_sigma": 2.8,
        "flags": 1,
    }

    optical_flow_channel0 = flow.optical_flow(arr=tiff_arr, channel=0)
    save_dir1 = tmp_path / "save_dir1"
    assert not save_dir1.exists()
    save.save_optical_flow_as_matlab(name1, optical_flow_channel0, save_dir1)
    optical_flow_channel0_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.mat", save_dir1
    )
    assert save_dir1.exists()
    assert optical_flow_channel0_path.exists()
    optical_flow_channel0_data = loadmat(optical_flow_channel0_path)
    optical_flow_channel0_arr = optical_flow_channel0_data["optical_flow"]
    assert isinstance(optical_flow_channel0_arr, np.ndarray)
    assert np.array_equal(optical_flow_channel0_arr, optical_flow_channel0)
    assert optical_flow_channel0_arr.shape == optical_flow_channel0.shape
    del optical_flow_channel0, optical_flow_channel0_data, optical_flow_channel0_arr
    gc.collect()
    shutil.rmtree(save_dir1)

    optical_flow_channel1 = flow.optical_flow(arr=tiff_arr, channel=1)
    save_dir2 = tmp_path / "save_dir2"
    assert not save_dir2.exists()
    save.save_optical_flow_as_matlab(name1, optical_flow_channel1, save_dir2)
    optical_flow_channel1_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.mat", save_dir2
    )
    assert save_dir2.exists()
    assert optical_flow_channel1_path.exists()
    optical_flow_channel1_data = loadmat(optical_flow_channel1_path)
    optical_flow_channel1_arr = optical_flow_channel1_data["optical_flow"]
    assert isinstance(optical_flow_channel1_arr, np.ndarray)
    assert np.array_equal(optical_flow_channel1_arr, optical_flow_channel1)
    assert optical_flow_channel1_arr.shape == optical_flow_channel1.shape
    del optical_flow_channel1, optical_flow_channel1_data, optical_flow_channel1_arr
    gc.collect()
    shutil.rmtree(save_dir2)

    optical_flow_channel2 = flow.optical_flow(arr=tiff_arr, channel=2)
    save_dir3 = tmp_path / "save_dir3"
    assert not save_dir3.exists()
    save.save_optical_flow_as_matlab(name1, optical_flow_channel2, save_dir3)
    optical_flow_channel2_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.mat", save_dir3
    )
    assert save_dir3.exists()
    assert optical_flow_channel2_path.exists()
    optical_flow_channel2_data = loadmat(optical_flow_channel2_path)
    optical_flow_channel2_arr = optical_flow_channel2_data["optical_flow"]
    assert isinstance(optical_flow_channel2_arr, np.ndarray)
    assert np.array_equal(optical_flow_channel2_arr, optical_flow_channel2)
    assert optical_flow_channel2_arr.shape == optical_flow_channel2.shape
    del optical_flow_channel2, optical_flow_channel2_data, optical_flow_channel2_arr
    gc.collect()
    shutil.rmtree(save_dir3)

    optical_flow_channel0_custom = flow.optical_flow(arr=tiff_arr, channel=0, **kwargs1)
    save_dir4 = tmp_path / "save_dir4"
    assert not save_dir4.exists()
    save.save_optical_flow_as_matlab(name1, optical_flow_channel0_custom, save_dir4)
    optical_flow_channel0_custom_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.mat", save_dir4
    )
    assert save_dir4.exists()
    assert optical_flow_channel0_custom_path.exists()
    optical_flow_channel0_custom_data = loadmat(optical_flow_channel0_custom_path)
    optical_flow_channel0_custom_arr = optical_flow_channel0_custom_data["optical_flow"]
    assert isinstance(optical_flow_channel0_custom_arr, np.ndarray)
    assert np.array_equal(
        optical_flow_channel0_custom_arr, optical_flow_channel0_custom
    )
    assert optical_flow_channel0_custom_arr.shape == optical_flow_channel0_custom.shape
    del (
        optical_flow_channel0_custom,
        optical_flow_channel0_custom_data,
        optical_flow_channel0_custom_arr,
    )
    gc.collect()
    shutil.rmtree(save_dir4)

    calculate_optical_flow = flow.calculate_optical_flow(arr=tiff_arr)
    save_dir5 = tmp_path / "save_dir5"
    assert not save_dir5.exists()
    save.save_optical_flow_as_matlab(name1, calculate_optical_flow, save_dir5)
    calculate_optical_flow_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.mat", save_dir5
    )
    assert save_dir5.exists()
    assert calculate_optical_flow_path.exists()
    calculate_optical_flow_data = loadmat(calculate_optical_flow_path)
    calculate_optical_flow_arr = calculate_optical_flow_data["optical_flow"]
    assert isinstance(calculate_optical_flow_arr, np.ndarray)
    assert np.array_equal(calculate_optical_flow_arr, calculate_optical_flow)
    assert calculate_optical_flow_arr.shape == calculate_optical_flow.shape
    del calculate_optical_flow, calculate_optical_flow_data, calculate_optical_flow_arr
    gc.collect()
    shutil.rmtree(save_dir5)

    calculate_optical_flow_default_true = flow.calculate_optical_flow(arr=tiff_arr)
    save_dir6 = tmp_path / "save_dir6"
    assert not save_dir6.exists()
    save.save_optical_flow_as_matlab(
        name1, calculate_optical_flow_default_true, save_dir6
    )
    calculate_optical_flow_default_true_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.mat", save_dir6
    )
    assert save_dir6.exists()
    assert calculate_optical_flow_default_true_path.exists()
    calculate_optical_flow_default_true_data = loadmat(
        calculate_optical_flow_default_true_path
    )
    calculate_optical_flow_default_true_arr = calculate_optical_flow_default_true_data[
        "optical_flow"
    ]
    assert isinstance(calculate_optical_flow_default_true_arr, np.ndarray)
    assert np.array_equal(
        calculate_optical_flow_default_true_arr, calculate_optical_flow_default_true
    )
    assert (
        calculate_optical_flow_default_true_arr.shape
        == calculate_optical_flow_default_true.shape
    )
    del (
        calculate_optical_flow_default_true,
        calculate_optical_flow_default_true_data,
        calculate_optical_flow_default_true_arr,
    )
    gc.collect()
    shutil.rmtree(save_dir6)


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

    name1 = "Test_Name"

    kwargs1 = {
        "pyr_scale": 0.25,
        "levels": 2,
        "winsize": 30,
        "iterations": 5,
        "poly_n": 7,
        "poly_sigma": 2.8,
        "flags": 1,
    }

    optical_flow_channel0 = flow.optical_flow(arr=tiff_arr, channel=0)
    save_dir1 = tmp_path / "save_dir1"
    assert not save_dir1.exists()
    save.save_optical_flow_as_numpy(name1, optical_flow_channel0, save_dir1)
    optical_flow_channel0_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.npy", save_dir1
    )
    assert save_dir1.exists()
    assert optical_flow_channel0_path.exists()
    optical_flow_channel0_arr = np.load(optical_flow_channel0_path)
    assert isinstance(optical_flow_channel0_arr, np.ndarray)
    assert np.array_equal(optical_flow_channel0_arr, optical_flow_channel0)
    assert optical_flow_channel0_arr.shape == optical_flow_channel0.shape
    del optical_flow_channel0, optical_flow_channel0_arr
    gc.collect()
    shutil.rmtree(save_dir1)

    optical_flow_channel1 = flow.optical_flow(arr=tiff_arr, channel=1)
    save_dir2 = tmp_path / "save_dir2"
    assert not save_dir2.exists()
    save.save_optical_flow_as_numpy(name1, optical_flow_channel1, save_dir2)
    optical_flow_channel1_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.npy", save_dir2
    )
    assert save_dir2.exists()
    assert optical_flow_channel1_path.exists()
    optical_flow_channel1_arr = np.load(optical_flow_channel1_path)
    assert isinstance(optical_flow_channel1_arr, np.ndarray)
    assert np.array_equal(optical_flow_channel1_arr, optical_flow_channel1)
    assert optical_flow_channel1_arr.shape == optical_flow_channel1.shape
    del optical_flow_channel1, optical_flow_channel1_arr
    gc.collect()
    shutil.rmtree(save_dir2)

    optical_flow_channel2 = flow.optical_flow(arr=tiff_arr, channel=2)
    save_dir3 = tmp_path / "save_dir3"
    assert not save_dir3.exists()
    save.save_optical_flow_as_numpy(name1, optical_flow_channel2, save_dir3)
    optical_flow_channel2_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.npy", save_dir3
    )
    assert save_dir3.exists()
    assert optical_flow_channel2_path.exists()
    optical_flow_channel2_arr = np.load(optical_flow_channel2_path)
    assert isinstance(optical_flow_channel2_arr, np.ndarray)
    assert np.array_equal(optical_flow_channel2_arr, optical_flow_channel2)
    assert optical_flow_channel2_arr.shape == optical_flow_channel2.shape
    del optical_flow_channel2, optical_flow_channel2_arr
    gc.collect()
    shutil.rmtree(save_dir3)

    optical_flow_channel0_custom = flow.optical_flow(arr=tiff_arr, channel=0, **kwargs1)
    save_dir4 = tmp_path / "save_dir4"
    assert not save_dir4.exists()
    save.save_optical_flow_as_numpy(name1, optical_flow_channel0_custom, save_dir4)
    optical_flow_channel0_custom_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.npy", save_dir4
    )
    assert save_dir4.exists()
    assert optical_flow_channel0_custom_path.exists()
    optical_flow_channel0_custom_arr = np.load(optical_flow_channel0_custom_path)
    assert isinstance(optical_flow_channel0_custom_arr, np.ndarray)
    assert np.array_equal(
        optical_flow_channel0_custom_arr, optical_flow_channel0_custom
    )
    assert optical_flow_channel0_custom_arr.shape == optical_flow_channel0_custom.shape
    del optical_flow_channel0_custom, optical_flow_channel0_custom_arr
    gc.collect()
    shutil.rmtree(save_dir4)

    calculate_optical_flow = flow.calculate_optical_flow(arr=tiff_arr)
    save_dir5 = tmp_path / "save_dir5"
    assert not save_dir5.exists()
    save.save_optical_flow_as_numpy(name1, calculate_optical_flow, save_dir5)
    calculate_optical_flow_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.npy", save_dir5
    )
    assert save_dir5.exists()
    assert calculate_optical_flow_path.exists()
    calculate_optical_flow_arr = np.load(calculate_optical_flow_path)
    assert isinstance(calculate_optical_flow_arr, np.ndarray)
    assert np.array_equal(calculate_optical_flow_arr, calculate_optical_flow)
    assert calculate_optical_flow_arr.shape == calculate_optical_flow.shape
    del calculate_optical_flow, calculate_optical_flow_arr
    gc.collect()
    shutil.rmtree(save_dir5)

    calculate_optical_flow_default_true = flow.calculate_optical_flow(arr=tiff_arr)
    save_dir6 = tmp_path / "save_dir6"
    assert not save_dir6.exists()
    save.save_optical_flow_as_numpy(
        name1, calculate_optical_flow_default_true, save_dir6
    )
    calculate_optical_flow_default_true_path = get_last_saved_pattern_fn_path(
        name1, lambda i: f"{name1}_flow{i}.npy", save_dir6
    )
    assert save_dir6.exists()
    assert calculate_optical_flow_default_true_path.exists()
    calculate_optical_flow_default_true_arr = np.load(
        calculate_optical_flow_default_true_path
    )
    assert isinstance(calculate_optical_flow_default_true_arr, np.ndarray)
    assert np.array_equal(
        calculate_optical_flow_default_true_arr, calculate_optical_flow_default_true
    )
    assert (
        calculate_optical_flow_default_true_arr.shape
        == calculate_optical_flow_default_true.shape
    )
    del calculate_optical_flow_default_true, calculate_optical_flow_default_true_arr
    gc.collect()
    shutil.rmtree(save_dir6)


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

    fig, ax = plt.subplots()

    preprocess_kwargs5 = {
        "gauss": {"ksize": (3, 3), "sigmaX": 1.5},
        "median": {"ksize": 3},
        "minmax": {"alpha": 0, "beta": 255, "norm_type": cv2.NORM_MINMAX},
        "contrast": {"alpha": 1.5, "beta": 20},
        "skip": [],
    }
    preprocess_kwargs6 = {
        "gauss": {"ksize": (7, 7)},
        "median": {"ksize": 9},
        "minmax": {},
        "contrast": {"alpha": 1.0},
        "skip": ["gauss", "median", "minmax", "contrast"],
    }
    preprocess_kwargs7 = {
        "gauss": {"sigmaX": 1.5},
        "minmax": {"alpha": 0, "beta": 1},
        "skip": ["minmax", "contrast"],
    }

    kwargs1 = {"T": 10, "fps": 20}
    kwargs2 = {"T": 40}
    kwargs3 = {"fps": 35}
    kwargs4 = {}
    kwargs5 = {"T": 1}

    stack1 = np.asarray(img.arr[:, 2, :, :])
    stack2 = np.asarray([img.arr[0, 0, :, :]])
    stack3 = np.asarray(
        [img.arr[0, 2, :, :], img.arr[(f - 1) // 2, 1, :, :], img.arr[f - 1, 0, :, :]]
    )
    stack4 = img.preprocess_stack(
        np.asarray(img.arr[: (f - 1) // 2, 2, :, :]), **preprocess_kwargs5
    )
    stack5 = img.preprocess_stack(np.asarray(img.arr[:, 0, :, :]), **preprocess_kwargs6)
    stack6 = img.preprocess_stack(np.asarray(img.arr[:, 1, :, :]), **preprocess_kwargs7)

    stack1_kwargs1_path = tmp_path / "stack1_kwargs1.mp4"
    stack1_kwargs3_path = tmp_path / "stack1_kwargs3.mp4"
    stack2_kwargs4_path = tmp_path / "stack2_kwargs4.mp4"
    stack2_kwargs5_path = tmp_path / "stack2_kwargs5.mp4"
    stack3_kwargs3_path = tmp_path / "stack3_kwargs3.mp4"
    stack3_kwargs5_path = tmp_path / "stack3_kwargs5.mp4"
    stack4_kwargs2_path = tmp_path / "stack4_kwargs2.mp4"
    stack4_kwargs4_path = tmp_path / "stack4_kwargs4.mp4"
    stack5_kwargs1_path = tmp_path / "stack5_kwargs1.mp4"
    stack5_kwargs3_path = tmp_path / "stack5_kwargs3.mp4"
    stack6_kwargs2_path = tmp_path / "stack6_kwargs2.mp4"
    stack6_kwargs4_path = tmp_path / "stack6_kwargs4.mp4"

    im_stack1 = ax.imshow(stack1[0], cmap="gray")
    im_stack2 = ax.imshow(stack2[0], cmap="gray")
    im_stack3 = ax.imshow(stack3[0], cmap="gray")
    im_stack4 = ax.imshow(stack4[0], cmap="gray")
    im_stack5 = ax.imshow(stack5[0], cmap="gray")
    im_stack6 = ax.imshow(stack6[0], cmap="gray")

    save_stack1_kwargs1 = save.save_original_video(
        "stack1_kwargs1", stack1_kwargs1_path, im_stack1, stack1, fig, ax, **kwargs1
    )
    save_stack1_kwargs3 = save.save_original_video(
        "stack1_kwargs3", stack1_kwargs3_path, im_stack1, stack1, fig, ax, **kwargs3
    )
    save_stack2_kwargs4 = save.save_original_video(
        "stack2_kwargs4", stack2_kwargs4_path, im_stack2, stack2, fig, ax, **kwargs4
    )
    save_stack2_kwargs5 = save.save_original_video(
        "stack2_kwargs2", stack2_kwargs5_path, im_stack2, stack2, fig, ax, **kwargs5
    )
    save_stack3_kwargs3 = save.save_original_video(
        "stack3_kwargs3", stack3_kwargs3_path, im_stack3, stack3, fig, ax, **kwargs3
    )
    save_stack3_kwargs5 = save.save_original_video(
        "stack3_kwargs1", stack3_kwargs5_path, im_stack3, stack3, fig, ax, **kwargs5
    )
    save_stack4_kwargs2 = save.save_original_video(
        "stack4_kwargs2", stack4_kwargs2_path, im_stack4, stack4, fig, ax, **kwargs2
    )
    save_stack4_kwargs4 = save.save_original_video(
        "stack4_kwargs4", stack4_kwargs4_path, im_stack4, stack4, fig, ax, **kwargs4
    )
    save_stack5_kwargs1 = save.save_original_video(
        "stack5_kwargs1", stack5_kwargs1_path, im_stack5, stack5, fig, ax, **kwargs1
    )
    save_stack5_kwargs3 = save.save_original_video(
        "stack5_kwargs3", stack5_kwargs3_path, im_stack5, stack5, fig, ax, **kwargs3
    )
    save_stack6_kwargs2 = save.save_original_video(
        "stack6_kwargs2", stack6_kwargs2_path, im_stack6, stack6, fig, ax, **kwargs2
    )
    save_stack6_kwargs4 = save.save_original_video(
        "stack6_kwargs4", stack6_kwargs4_path, im_stack6, stack6, fig, ax, **kwargs4
    )

    assert stack1_kwargs1_path.exists()
    assert stack1_kwargs3_path.exists()
    assert stack2_kwargs4_path.exists()
    assert stack2_kwargs5_path.exists()
    assert stack3_kwargs3_path.exists()
    assert stack3_kwargs5_path.exists()
    assert stack4_kwargs2_path.exists()
    assert stack4_kwargs4_path.exists()
    assert stack5_kwargs1_path.exists()
    assert stack5_kwargs3_path.exists()
    assert stack6_kwargs2_path.exists()
    assert stack6_kwargs4_path.exists()

    assert os.path.getsize(stack1_kwargs1_path) > 0
    assert os.path.getsize(stack1_kwargs3_path) > 0
    assert os.path.getsize(stack2_kwargs4_path) > 0
    assert os.path.getsize(stack2_kwargs5_path) > 0
    assert os.path.getsize(stack3_kwargs3_path) > 0
    assert os.path.getsize(stack3_kwargs5_path) > 0
    assert os.path.getsize(stack4_kwargs2_path) > 0
    assert os.path.getsize(stack4_kwargs4_path) > 0
    assert os.path.getsize(stack5_kwargs1_path) > 0
    assert os.path.getsize(stack5_kwargs3_path) > 0
    assert os.path.getsize(stack6_kwargs2_path) > 0
    assert os.path.getsize(stack6_kwargs4_path) > 0