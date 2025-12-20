import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
import yaml
from src.cell_tracking import memory
from src.cell_tracking.defaults import default_yaml_config
from dataclasses import dataclass


@dataclass
class MMFixture:
    """
    Dataclass to hold MemoryManager fixture data.
    """

    MM: memory.MemoryManager
    tmp_path: str
    main_path: str
    yaml_path: str


@pytest.fixture
def sample_MemoryManager(tmp_path):
    """
    Fixture to provide a sample MemoryManager instance for testing.

    Args:
        tmp_path (str): Temporary path provided by pytest.

    Returns:
        MMFixture: A dataclass containing the MemoryManager instance and related paths.
    """
    return MMFixture(
        MM=memory.MemoryManager(tmp_path),
        tmp_path=tmp_path,
        main_path=os.path.join(tmp_path, "Cell-Tracking"),
        yaml_path=os.path.join(tmp_path, "Cell-Tracking/config.yaml"),
    )


def test_init(sample_MemoryManager):
    """
    Tests the initialization of the MemoryManager class.

    Args:
        sample_MemoryManager (MMFixture): A dataclass containing the MemoryManager instance and related paths.

    Returns:
        None.
    """
    f = sample_MemoryManager

    assert hasattr(f.MM, "path")
    assert hasattr(f.MM, "yaml_path")
    assert hasattr(f.MM, "config")

    assert f.MM.path is not None
    assert f.MM.yaml_path is not None
    assert f.MM.config is not None


def test_write_default_yaml(sample_MemoryManager):
    """
    Tests whether the default YAML configuration is written correctly.

    Args:
        sample_MemoryManager (MMFixture): A dataclass containing the MemoryManager instance and related paths.

    Returns:
        None.
    """
    f = sample_MemoryManager

    assert f.MM.config == default_yaml_config

    with open(f.MM.yaml_path) as y:
        assert yaml.safe_load(y) == default_yaml_config


def test_create_main_dir(sample_MemoryManager):
    """
    Tests whether the main directory and YAML file are created correctly.

    Args:
        sample_MemoryManager (MMFixture): A dataclass containing the MemoryManager instance and related paths.

    Returns:
        None.
    """
    f = sample_MemoryManager
    assert os.path.exists(f.MM.path)
    assert os.path.exists(f.MM.yaml_path)


def test_read_yaml(sample_MemoryManager):
    """
    Tests the reading of the YAML configuration file.

    Args:
        sample_MemoryManager (MMFixture): A dataclass containing the MemoryManager instance and related paths.

    Returns:
        None.
    """
    f = sample_MemoryManager
    f.MM.read_yaml()
    assert f.MM.config == default_yaml_config

    f = sample_MemoryManager
    custom_yaml_config = {
        "preprocess_args": {
            "gauss": {"ksize": [7, 7], "sigmaX": 2.0},
            "median": {"ksize": 3},
            "normalize": {"alpha": 0, "beta": 255},
            "contrast": {"alpha": 1.5, "beta": 10.0},
            "skip": [],
        },
        "farneback_args": {
            "levels": 1.0,
            "winsize": 5,
            "iterations": 5,
            "poly_n": 7,
            "poly_sigma": 1.5,
            "flags": 1,
        },
        "raft_args": {
            "model_size": 2,
            "model_weights_path": "/tmp/model.pth",
            "gpu_flag": True,
        },
    }

    with open(f.MM.yaml_path, "w") as y:
        yaml.dump(custom_yaml_config, y)
    f.MM.read_yaml()
    assert f.MM.config == custom_yaml_config


def test_create_tiff_dir(sample_MemoryManager):
    """
    Tests whether the TIFF directory and its subdirectories are created correctly.

    Args:
        sample_MemoryManager (MMFixture): A dataclass containing the MemoryManager instance and related paths.

    Returns:
        None.
    """
    f = sample_MemoryManager
    tiff_name = "tiff1"

    f.MM.create_tiff_dir(tiff_name)

    parent_path = os.path.join(f.MM.path, tiff_name)
    assert os.path.exists(parent_path)

    subdirs = ["raw_data", "optical_flows", "heatmaps", "kymographs"]

    for sub in subdirs:
        assert os.path.exists(os.path.join(parent_path, sub))
