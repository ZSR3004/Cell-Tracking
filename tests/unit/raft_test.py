import torch
import pytest
import numpy as np
from src import raft
from src import tiffclass as tiff


@pytest.fixture
def init_tiff():
    original_path = (
        "../../datasets/nuclei_labeled/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif"
    )
    return tiff.Tiff(original_path)


### pad_to_multiple_of_8 ###
def _check_divisible_by_8():
    raise NotImplementedError


def _check_preserved_values():
    raise NotImplementedError


def _check_padding_behavior():
    raise NotImplementedError


def _check_no_change_over_device():
    raise NotImplementedError


def _check_idempotence():
    raise NotImplementedError


def test_pad_to_multiple_of_8():
    raise NotImplementedError


### preprocess_tensor ###
def test_preprocess_tensor():
    raise NotImplementedError


### batch_frames ###
def test_batch_frames():
    raise NotImplementedError


### get_raft_optical_flow ###
def test_get_raft_optical_flow():
    raise NotImplementedError


### make_raft_output_array ###
def test_make_raft_output_array():
    raise NotImplementedError


### calcOpticalFlowRAFT ###
def test_calcOpticalFlowRAFT():
    raise NotImplementedError
