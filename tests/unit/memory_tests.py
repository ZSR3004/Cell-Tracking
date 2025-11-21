import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
import yaml
from cell_tracking import memory
from cell_tracking.defaults import default_yaml_config
from dataclasses import dataclass


@dataclass
class MMFixture:
    MM: memory.MemoryManager
    tmp_path: str
    main_path: str
    yaml_path: str


@pytest.fixture
def sample_MemoryManager(tmp_path):
    return MMFixture(
        MM=memory.MemoryManager(tmp_path),
        tmp_path=tmp_path,
        main_path=os.path.join(tmp_path, "Cell-Tracking"),
        yaml_path=os.path.join(tmp_path, "Cell-Tracking/config.yaml"),
    )


def test_init(sample_MemoryManager):
    f = sample_MemoryManager

    assert f.MM.path == f.main_path
    assert f.MM.yaml_path == f.yaml_path
    assert f.MM.config == default_yaml_config


def test_write_default_yaml(sample_MemoryManager):
    f = sample_MemoryManager
    with open(f.yaml_path) as y:
        assert yaml.safe_load(y) == default_yaml_config


def test_create_main_dir(sample_MemoryManager):
    f = sample_MemoryManager
    assert os.path.exists(f.main_path)
    assert os.path.exists(f.yaml_path)


def test_read_yaml(sample_MemoryManager):
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
    f = sample_MemoryManager
    tiff_name = "tiff1"

    f.MM.create_tiff_dir(tiff_name)

    parent_path = os.path.join(f.main_path, tiff_name)
    assert os.path.exists(parent_path)

    subdirs = ["raw_data", "optical_flows", "heatmaps", "kymographs"]

    for sub in subdirs:
        assert os.path.exists(os.path.join(parent_path, sub))
