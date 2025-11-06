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
    """
    Creates a Tiff class instance.

    Args:
        request (pytest.FixtureRequest): The paths to generate Tiff instances from.

    Returns:
        (tiff.Tiff): Tiff class instance of the path.
    """
    return tiff.Tiff(request.param)


@pytest.fixture(params=TIFF_PATHS)
def init_torch_tensor(request: pytest.FixtureRequest) -> torch.Tensor:
    """
    Extracts the array from a Tiff class instance and generates a tensor from it
    that is ready for preprocessing.

    Args:
        request (pytest.FixtureRequest): The paths to generate Tiff instances from.

    Returns:
        (torch.Tensor): tensor representation of the tiff file.
    """
    tiff_file = tiff.Tiff(request.param)
    arr = tiff_file.arr[:, 0, ...]
    return torch.from_numpy(arr).unsqueeze(1).repeat(1, 3, 1, 1)


class TensorHelpers:
    def _check_if_float32_tensor(self, ten: torch.Tensor) -> None:
        """
        Checks if the tensor is of type float32.
        Args:
            ten (torch.Tensor): tensor to check.
        """
        assert ten.dtype == torch.float32

    def _check_shape(self, ten: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
        """
        Checks if the tensor has the expected shape.

        Args:
            ten (torch.Tensor): tensor to check.
            expected_shape (tuple[int, ...]): expected shape of the tensor.
        """
        assert ten.shape == expected_shape

    def _check_normalization(self, ten: torch.Tensor) -> None:
        """
        Checks if the tensor values are within the range [0, 255].

        Args:
            ten (torch.Tensor): tensor to check.
        """
        assert torch.all((ten >= 0) & (ten <= 255))


class NdarrayHelpers:
    def _check_if_float32_np(self, arr: np.ndarray) -> None:
        """
        Checks if the numpy array is of type float32.

        Args:
            arr (np.ndarray): array to check.
        """
        assert arr.dtype == np.float32

    def _check_shape_np(self, arr: np.ndarray, expected_shape: tuple[int, ...]) -> None:
        """
        Checks if the numpy array has the expected shape.

        Args:
            arr (np.ndarray): array to check.
            expected_shape (tuple[int, ...]): expected shape of the array.
        """
        assert arr.shape == expected_shape

    def _check_normalization_np(self, arr: np.ndarray) -> None:
        """
        Checks if the numpy array values are within the range [0, 255].

        Args:
            arr (np.ndarray): array to check.
        """
        assert np.all((arr >= 0) & (arr <= 255))


class TestPadToMultipleOf8(TensorHelpers):
    def _check_divisible_by_8(self, ten: torch.Tensor) -> None:
        """
        Checks if all dimensions of the tensor are divisible by 8.

        Args:
            ten (torch.Tensor): tensor to check.
        """
        shape = ten.shape
        for dim in shape:
            assert dim % 8 == 0

    def _check_preserved_values(self, ten: torch.Tensor, pad_ten: torch.Tensor) -> None:
        """
        Checks if the original tensor values are preserved in the padded tensor.

        Args:
            ten (torch.Tensor): original tensor.
            pad_ten (torch.Tensor): padded tensor.
        """
        og_shape = ten.shape
        pad_ten_og_dims = pad_ten[og_shape[0], og_shape[1], og_shape[2], og_shape[3]]
        assert torch.equal(ten, pad_ten_og_dims)

    def _check_idempotence(self, ten: torch.Tensor) -> None:
        """
        Checks if padding an already padded tensor does not change it.

        Args:
            ten (torch.Tensor): padded tensor.
        """
        ten2 = raft.pad_to_multiple_of_8(ten)
        ten3 = raft.pad_to_multiple_of_8(ten)
        ten4 = raft.pad_to_multiple_of_8(ten)

        assert torch.equal(ten, ten2)
        assert torch.equal(ten, ten3)
        assert torch.equal(ten, ten4)

    def test_pad_to_multiple_of_8(self, init_torch_tensor: torch.Tensor) -> None:
        """
        Tests the pad_to_multiple_of_8 function.
        """
        ten = init_torch_tensor
        pad_ten = raft.pad_to_multiple_of_8(ten)

        self._check_divisible_by_8(pad_ten)
        self._check_preserved_values(ten, pad_ten)
        self._check_idempotence(pad_ten)


class TestPreprocessTensor(TensorHelpers):
    def test_preprocess_tensor(
        self, init_tiff: tiff.Tiff, init_torch_tensor: torch.Tensor
    ) -> None:
        """
        Tests the preprocess_tensor function.
        """
        tiff_file = init_tiff
        ten = init_torch_tensor
        pad_ten = raft.pad_to_multiple_of_8(ten)
        expected_shape = pad_ten.shape

        ten = raft.preprocess_tensor(tiff_file)

        self._check_if_float32_tensor(ten)
        self._check_shape(ten, expected_shape)
        self._check_normalization(ten)


class TestBatchFrames(TensorHelpers):
    def test_batch_frames(self, init_torch_tensor: torch.Tensor) -> None:
        """
        Tests the batch_frames function.
        """
        ten = init_torch_tensor
        batch1, batch2 = raft.batch_frames(ten)

        assert batch1.shape == batch2.shape
        assert batch1[0] == ten[0]
        assert batch1[-1] == ten[-2]

        assert batch2[0] == ten[1]
        assert batch2[-1] == ten[-1]


class TestGetRAFTOpticalFlow(TensorHelpers):
    def _check_model_size_logic(
        self,
        ten: torch.Tensor,
    ) -> None:
        """
        Checks if the RAFT model size selection logic is correct.

        Args:
            ten (torch.Tensor): tensor to check.
        """
        raise NotImplementedError

    def _check_if_custom_weights_used(self, ten: torch.Tensor) -> None:
        """
        Checks if custom weights are used in the RAFT model.

        Args:
            ten (torch.Tensor): tensor to check.
        """
        raise NotImplementedError

    def _check_if_gpu_used(self, ten: torch.Tensor) -> None:
        """
        Checks if GPU is used in the RAFT model.

        Args:
            ten (torch.Tensor): tensor to check.
        """
        raise NotImplementedError

    def test_get_raft_optical_flow(self, init_tiff: tiff.Tiff) -> None:
        """
        Tests the get_raft_optical_flow function.
        """
        raise NotImplementedError


class TestMakeRAFTOutputArray(NdarrayHelpers):
    def test_make_raft_output_array(
        self, init_tiff: tiff.Tiff, init_torch_tensor: torch.Tensor
    ) -> None:
        """
        Tests the make_raft_output_array function.
        """
        tiff_file = init_tiff
        ten = init_torch_tensor
        flow = raft.get_raft_optical_flow(tiff_file, use_gpu=True)
        flow_arr = raft.make_raft_output_array(flow)

        self._check_if_float32_np(flow_arr)

        expected_shape = (ten.shape[0] - 1, ten.shape[1], ten.shape[2], 2)
        self._check_shape_np(flow_arr, expected_shape)


class TestCalcOpticalFlowRAFT:
    def test_calcOpticalFlowRAFT(self, init_tiff: tiff.Tiff) -> None:
        """
        Tests the calcOpticalFlowRAFT function.
        """
        raise NotImplementedError
