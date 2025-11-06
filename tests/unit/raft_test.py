import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import pytest
import numpy as np
from src import tiffclass as tiff
from src import raft

TIFF_PATHS = [
    "../../datasets/nuclei_labeled/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif"
]

@pytest.fixture(params=TIFF_PATHS)
def init_tiff(request: pytest.FixtureRequest) -> tiff.Tiff:
    return tiff.Tiff(request.param)

@pytest.fixture(params=TIFF_PATHS)
def init_torch_tensor(request: pytest.FixtureRequest) -> torch.Tensor:
    tiff_file = tiff.Tiff(request.param)
    arr = tiff_file.arr[:, 0, ...]
    return torch.from_numpy(arr).unsqueeze(1).repeat(1, 3, 1, 1)

class TensorHelpers:
    def _check_if_float32_tensor(self, ten: torch.Tensor) -> None:
        assert ten.dtype == torch.float32

    def _check_shape(self, ten: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
        assert ten.shape == expected_shape

    def _check_normalization(self, ten: torch.Tensor) -> None:
        assert torch.all((ten >= 0) & (ten <= 255))


class NdarrayHelpers:
    def _check_if_float32_np(self, arr: np.ndarray) -> None:
        assert arr.dtype == np.float32

    def _check_shape_np(self, arr: np.ndarray, expected_shape: tuple[int, ...]) -> None:
        assert arr.shape == expected_shape

    def _check_normalization_np(self, arr: np.ndarray) -> None:
        assert np.all((arr >= 0) & (arr <= 255))


class TestPadToMultipleOf8(TensorHelpers):
    def _check_divisible_by_8(self, ten: torch.Tensor) -> None:
        shape = ten.shape
        for dim in shape:
            assert dim % 8 == 0

    def _check_preserved_values(self, ten: torch.Tensor, pad_ten: torch.Tensor) -> None:
        og_shape = ten.shape 
        pad_ten_og_dims = pad_ten[og_shape[0], og_shape[1], og_shape[2], og_shape[3]]
        assert torch.equal(ten, pad_ten_og_dims)

    def _check_idempotence(self, ten: torch.Tensor) -> None:
        ten2 = raft.pad_to_multiple_of_8(ten)
        ten3 = raft.pad_to_multiple_of_8(ten)
        ten4 = raft.pad_to_multiple_of_8(ten)

        assert torch.equal(ten, ten2)
        assert torch.equal(ten, ten3)
        assert torch.equal(ten, ten4)

    def test_pad_to_multiple_of_8(self, init_torch_tensor: torch.Tensor) -> None:
        ten = init_torch_tensor
        pad_ten = raft.pad_to_multiple_of_8(ten)

        self._check_divisible_by_8(pad_ten)
        self._check_preserved_values(ten, pad_ten)
        self._check_idempotence(pad_ten)


class TestPreprocessTensor(TensorHelpers):
    def test_preprocess_tensor(self, init_tiff: tiff.Tiff) -> None:
        raise NotImplementedError


class TestBatchFrames(TensorHelpers):
    def test_batch_frames(self, init_torch_tensor: torch.Tensor) -> None:
        raise NotImplementedError


class TestGetRAFTOpticalFlow(TensorHelpers):
    def _check_model_size_logic(
        self,
        ten: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def _check_if_custom_weights_used(self, ten: torch.Tensor) -> None:
        raise NotImplementedError

    def _check_if_gpu_used(self, ten: torch.Tensor) -> None:
        raise NotImplementedError

    def test_get_raft_optical_flow(self, init_tiff: tiff.Tiff) -> None:
        raise NotImplementedError


class TestMakeRAFTOutputArray(NdarrayHelpers):
    def test_make_raft_output_array(self, init_tiff: tiff.Tiff) -> None:
        raise NotImplementedError


class TestCalcOpticalFlowRAFT:
    def test_calcOpticalFlowRAFT(self, init_tiff: tiff.Tiff) -> None:
        raise NotImplementedError
