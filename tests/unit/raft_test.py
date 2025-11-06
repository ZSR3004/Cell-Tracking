import torch
import pytest
import numpy as np
from src import raft
from src import tiffclass as tiff

TIFF_PATHS = [
    "../../datasets/nuclei_labeled/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED.tif"
]


@pytest.fixture(params=TIFF_PATHS)
def init_tiff(request: pytest.FixtureRequest) -> tiff.Tiff:
    return tiff.Tiff(request.param)


class TensorHelpers:
    def _check_if_float32_tensor(self, ten: torch.Tensor) -> None:
        raise NotImplementedError

    def _check_shape(self, ten: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
        raise NotImplementedError

    def _check_normalization(self, ten: torch.Tensor) -> None:
        raise NotImplementedError


class NdarrayHelpers:
    def _check_if_float32_np(self, arr: np.ndarray) -> None:
        raise NotImplementedError

    def _check_shape_np(self, arr: np.ndarray, expected_shape: tuple[int, ...]) -> None:
        raise NotImplementedError

    def _check_normalization_np(self, arr: np.ndarray) -> None:
        raise NotImplementedError


class TestPadToMultipleOf8(TensorHelpers):
    def _check_divisible_by_8(self, ten: torch.Tensor) -> None:
        raise NotImplementedError

    def _check_preserved_values(self, ten: torch.Tensor, pad_ten: torch.Tensor) -> None:
        raise NotImplementedError

    def _check_idempotence(self, ten: torch.Tensor) -> None:
        raise NotImplementedError

    def test_pad_to_multiple_of_8(self, init_tiff: tiff.Tiff) -> None:
        raise NotImplementedError


class TestPreprocessTensor(TensorHelpers):
    def test_preprocess_tensor(self, init_tiff: tiff.Tiff) -> None:
        raise NotImplementedError


class TestBatchFrames(TensorHelpers):
    def test_batch_frames(self, init_tiff: tiff.Tiff) -> None:
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


class TestMakeRAFTOutputArray(ndarrayHelpers):
    def test_make_raft_output_array(self, init_tiff: tiff.Tiff) -> None:
        raise NotImplementedError


class TestCalcOpticalFlowRAFT:
    def test_calcOpticalFlowRAFT(self, init_tiff: tiff.Tiff) -> None:
        raise NotImplementedError
