import torch
import pytest
import numpy as np
from src import raft
from src import tiffclass as tiff

@pytest.fixture
def init_tiff():
    original_path = "../../datasets/nuclei_labeled/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif"
    return tiff.Tiff(original_path)

def test_pad_to_multiple_of_8():
    raise NotImplementedError

def test_preprocess_tensor():
    raise NotImplementedError

def test_batch_frames():
    raise NotImplementedError

def test_get_raft_optical_flow():
    raise NotImplementedError

def test_make_raft_output_array():
    raise NotImplementedError

def test_calcOpticalFlowRAFT():
    raise NotImplementedError