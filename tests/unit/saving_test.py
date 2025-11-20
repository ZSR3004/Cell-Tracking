import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
   
import cv2, json, pytest
from src import tiffclass as tiff
from src import saving
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.animation as animation
from src.defaults import default_process, default_flow, default_trajectory

TIFF_PATHS = [
    ("../../datasets/nuclei_labeled/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif", (96, 3, 520, 2329))
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
    return tiff.Tiff(request.param)

def test_get_unique_path(init_tiff: tiff.Tiff, tmp_path):
    """
    Tests whether the get_unique_path method works correctly.

    Args:
        init_tiff (tiff.Tiff): A tuple containing information about the TIFF file:
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
        tmp_path (pathlib.Path): A path to a temporary directory (this is a fixture in Pytest).

    Return:
        None
    """
    path, f, c, h, w = init_tiff

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

    unique_path_npy_fn_1 = saving.get_unique_path(name1, npy_fn_1, tmp_path)
    assert save_dir1.exists()
    assert unique_path_npy_fn_1.name == "flow_flow1.npy"
    assert unique_path_npy_fn_1.parent == save_dir1

    unique_path_xyz_fn_2 = saving.get_unique_path(name2, xyz_fn_2, tmp_path)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert unique_path_xyz_fn_2.name == "Test_Name_flow1.xyz"
    assert unique_path_xyz_fn_2.parent == save_dir2

    unique_path_mat_fn_3 = saving.get_unique_path(name3, mat_fn_3, tmp_path)
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

    unique_path_npy_fn_2 = saving.get_unique_path(name2, npy_fn_2, tmp_path)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_npy_fn_2.name == "Test_Name_flow1.npy"
    assert unique_path_npy_fn_2.parent == save_dir2

    unique_path_xyz_fn_3 = saving.get_unique_path(name3, xyz_fn_3, tmp_path)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_xyz_fn_3.name == "1Name_flow1.xyz"
    assert unique_path_xyz_fn_3.parent == save_dir3

    unique_path_mat_fn_1 = saving.get_unique_path(name1, mat_fn_1, tmp_path)
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

    unique_path_npy_fn_3 = saving.get_unique_path(name3, npy_fn_3, tmp_path)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_npy_fn_3.name == "1Name_flow2.npy"
    assert unique_path_npy_fn_3.parent == save_dir3

    unique_path_xyz_fn_1 = saving.get_unique_path(name1, xyz_fn_1, tmp_path)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_xyz_fn_1.name == "flow_flow5.xyz"
    assert unique_path_xyz_fn_1.parent == save_dir1

    unique_path_mat_fn_2 = saving.get_unique_path(name2, mat_fn_2, tmp_path)
    assert save_dir1.exists()
    assert save_dir2.exists()
    assert save_dir3.exists()
    assert unique_path_mat_fn_2.name == "Test_Name_flow9.mat"
    assert unique_path_mat_fn_2.parent == save_dir2

def test_save_arr(init_tiff (tiff.Tiff), tmp_path):
    """
    Tests whether the save_arr method works correctly.

    Args:
        init_tiff (tiff.Tiff): A tuple containing information about the TIFF file:
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
        tmp_path (pathlib.Path): A path to a temporary directory (this is a fixture in Pytest).

    Return:
        None
    """
    path, f, c, h, w = init_tiff

    name1 = "Test_Name"

    tiff_instance = tiff(path)

    save_dir = tmp_path / name1
    assert not save_dir.exists()

    save_arr1 = saving.save_arr(name1, tiff_instance,)


    return NotImplementedError

def test_save_optical_flow_as_xyz():
    return NotImplementedError

def test_save_optical_flow_as_matlab():
    return NotImplementedError

def test_save_optical_flow_as_numpy():
    return NotImplementedError

#IMPORTANT: TO TEST save_optical_flow_as_xyz, TEST IF RESHAPING WORKS!!! FOR EXAMPLE, MAKE SAMPLE ARRAYS AND SEE IF THEY RESHAPE CORRECTLY.
#AN EXAMPLE OF A CORRECT RESHAPE WOULD BE (NOTE BOTH OF THEM ARE NUMPY ARRAYS): Original: [[[[[1, 2]],[[3, 4]],[[5, 6]]]],[[[[7, 8]],[[9, 10]],[[11, 12]]]]]. After reshaping: [[1, 2],[3, 4],[5, 6],[7, 8],[9, 10],[11, 12]] 
#   PROB NOT THIS BC NUMPY AUTOMATICALLY CONVERTS TUPLES TO LISTS (or instead is it this because they're tuples?) Original: [[[[(1, 2)],[(3, 4)],[(5, 6)]]],[[[(7, 8)],[(9, 10)],[(11, 12)]]]]. After reshaping: [(1, 2),(3, 4),(5, 6),(7, 8),(9, 10),(11, 12)] 
#ALSO TO TEST save_optical_flow_as_xyz, AFTER RESHAPING YOU SHOULD TEST IF DATATYPES OF ELEMENTS OF THE RESHAPED ARRAY ARE THE SAME DATATYPE AS THE ELEMENTS OF THE NON-RESHAPED ARRAY. LIKE FOR EXAMPLE THEY'RE ALL INTS

def test_save_original_video(init_tiff: tiff.Tiff, tmp_path):
    """
    Tests whether the save_original_video method works correctly.

    Args:
        init_tiff (tiff.Tiff): A tuple containing information about the TIFF file:
            - path (str): The path to the TIFF file.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
        tmp_path (pathlib.Path): A path to a temporary directory (this is a fixture in Pytest).

    Return:
        None
    """
    path, f, c, h, w = init_tiff
    img = tiff.Tiff(path)

    fig, ax = plt.subplots()

    preprocess_kwargs5 = {"gauss": {"ksize": (3, 3), "sigmaX": 1.5}, "median": {"ksize": 3}, "minmax": {"alpha": 0, "beta": 255, "norm_type": cv2.NORM_MINMAX}, "contrast": {"alpha": 1.5, "beta": 20}, "skip": []}
    preprocess_kwargs6 = {"gauss": {"ksize": (7, 7)}, "median": {"ksize": 9}, "minmax": {}, "contrast": {"alpha": 1.0}, "skip": ["gauss", "median", "minmax", "contrast"]}
    preprocess_kwargs7 = {"gauss": {"sigmaX": 1.5}, "minmax": {"alpha": 0, "beta": 1}, "skip": ["minmax", "contrast"]}

    kwargs1 = {'T': 10, 'fps': 20}
    kwargs2 = {'T': 40}
    kwargs3 = {'fps': 35}
    kwargs4 = {}
    kwargs5 = {'T': 1}

    stack1 = np.asarray(img.arr[:, 2, :, :])
    stack2 = np.asarray([img.arr[0, 0, :, :]])
    stack3 = np.asarray([img.arr[0, 2, :, :], img.arr[(f-1)//2, 1, :, :], img.arr[f-1, 0, :, :]])
    stack4 = img.preprocess_stack(np.asarray(img.arr[:(f-1)//2, 2, :, :]), **preprocess_kwargs5)
    stack5 = img.preprocess_stack(np.asarray(img.arr[:, 0, :, :]), **preprocess_kwargs6)
    stack6 = img.preprocess_stack(np.asarray(img.arr[:, 1, :, :]), **preprocess_kwargs7)

    stack1_kwargs1_path = tmp_path / 'stack1_kwargs1.mp4'
    stack1_kwargs3_path = tmp_path / 'stack1_kwargs3.mp4'
    stack2_kwargs4_path = tmp_path / 'stack2_kwargs4.mp4'
    stack2_kwargs5_path = tmp_path / 'stack2_kwargs5.mp4' 
    stack3_kwargs3_path = tmp_path / 'stack3_kwargs3.mp4'
    stack3_kwargs5_path = tmp_path / 'stack3_kwargs5.mp4'
    stack4_kwargs2_path = tmp_path / 'stack4_kwargs2.mp4'
    stack4_kwargs4_path = tmp_path / 'stack4_kwargs4.mp4'
    stack5_kwargs1_path = tmp_path / 'stack5_kwargs1.mp4'
    stack5_kwargs3_path = tmp_path / 'stack5_kwargs3.mp4'
    stack6_kwargs2_path = tmp_path / 'stack6_kwargs2.mp4'
    stack6_kwargs4_path = tmp_path / 'stack6_kwargs4.mp4'

    im_stack1 = ax.imshow(stack1[0], cmap='gray')
    im_stack2 = ax.imshow(stack2[0], cmap='gray')
    im_stack3 = ax.imshow(stack3[0], cmap='gray')
    im_stack4 = ax.imshow(stack4[0], cmap='gray')
    im_stack5 = ax.imshow(stack5[0], cmap='gray')
    im_stack6 = ax.imshow(stack6[0], cmap='gray')

    save_stack1_kwargs1 = saving.save_original_video("stack1_kwargs1", stack1_kwargs1_path, im_stack1, stack1, fig, ax, **kwargs1)
    save_stack1_kwargs3 = saving.save_original_video("stack1_kwargs3", stack1_kwargs3_path, im_stack1, stack1, fig, ax, **kwargs3)
    save_stack2_kwargs4 = saving.save_original_video("stack2_kwargs4", stack2_kwargs4_path, im_stack2, stack2, fig, ax, **kwargs4)
    save_stack2_kwargs5 = saving.save_original_video("stack2_kwargs2", stack2_kwargs5_path, im_stack2, stack2, fig, ax, **kwargs5)
    save_stack3_kwargs3 = saving.save_original_video("stack3_kwargs3", stack3_kwargs3_path, im_stack3, stack3, fig, ax, **kwargs3)
    save_stack3_kwargs5 = saving.save_original_video("stack3_kwargs1", stack3_kwargs5_path, im_stack3, stack3, fig, ax, **kwargs5)
    save_stack4_kwargs2 = saving.save_original_video("stack4_kwargs2", stack4_kwargs2_path, im_stack4, stack4, fig, ax, **kwargs2)
    save_stack4_kwargs4 = saving.save_original_video("stack4_kwargs4", stack4_kwargs4_path, im_stack4, stack4, fig, ax, **kwargs4)
    save_stack5_kwargs1 = saving.save_original_video("stack5_kwargs1", stack5_kwargs1_path, im_stack5, stack5, fig, ax, **kwargs1)
    save_stack5_kwargs3 = saving.save_original_video("stack5_kwargs3", stack5_kwargs3_path, im_stack5, stack5, fig, ax, **kwargs3)
    save_stack6_kwargs2 = saving.save_original_video("stack6_kwargs2", stack6_kwargs2_path, im_stack6, stack6, fig, ax, **kwargs2)
    save_stack6_kwargs4 = saving.save_original_video("stack6_kwargs4", stack6_kwargs4_path, im_stack6, stack6, fig, ax, **kwargs4)

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