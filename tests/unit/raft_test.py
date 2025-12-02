import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import pytest
import numpy as np
from src.cell_tracking import raft
from src.cell_tracking import tiffclass as tiff
from unittest.mock import Mock, patch

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
        (tiff.Tiff): A tuple containing information about the TIFF file.
            - img (str): A TIFF instance.
            - f (int): Number of frames.
            - c (int): Number of channels.
            - h (int): Height.
            - w (int): Width.
    """
    path, info = request.param
    return (tiff.Tiff(path), info)


@pytest.fixture(params=TIFF_PATHS)
def init_torch_tensor(request: pytest.FixtureRequest) -> torch.Tensor:
    """
    Extracts the array from a Tiff class instance and generates a tensor from it
    that is ready for preprocessing.

    Args:
        request (pytest.FixtureRequest): The paths to generate Tiff instances from.

    Returns:
        (torch.Tensor): Tensor representation of the tiff file.
    """
    path, info = request.param
    tiff_file = tiff.Tiff(path)
    arr = tiff_file.arr[:, 0, ...]
    original_dtype = arr.dtype
    arr = arr.astype("float32") / np.iinfo(original_dtype).max
    return torch.from_numpy(arr).unsqueeze(1).repeat(1, 3, 1, 1)


class TensorHelpers:
    def _check_if_float32_tensor(self, ten: torch.Tensor) -> None:
        """
        Checks if the tensor is of type float32.

        Args:
            ten (torch.Tensor): Tensor to check.

        Returns:
            None.
        """
        assert ten.dtype == torch.float32

    def _check_shape(self, ten: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
        """
        Checks if the tensor has the expected shape.

        Args:
            ten (torch.Tensor): Tensor to check.
            expected_shape (tuple[int, ...]): Expected shape of the tensor.

        Returns:
            None.
        """
        assert ten.shape == expected_shape

    def _check_normalization(self, ten: torch.Tensor) -> None:
        """
        Checks if the tensor values are within the range [0, 255].

        Args:
            ten (torch.Tensor): Tensor to check.

        Returns:
            None.
        """
        assert torch.all((ten >= 0) & (ten <= 255))


class NdarrayHelpers:
    def _check_if_float32_np(self, arr: np.ndarray) -> None:
        """
        Checks if the numpy array is of type float32.

        Args:
            arr (np.ndarray): Array to check.

        Returns:
            None.
        """
        assert arr.dtype == np.float32

    def _check_shape_np(self, arr: np.ndarray, expected_shape: tuple[int, ...]) -> None:
        """
        Checks if the numpy array has the expected shape.

        Args:
            arr (np.ndarray): Array to check.
            expected_shape (tuple[int, ...]): Expected shape of the array.

        Returns:
            None.
        """
        assert arr.shape == expected_shape

    def _check_normalization_np(self, arr: np.ndarray) -> None:
        """
        Checks if the numpy array values are within the range [0, 255].

        Args:
            arr (np.ndarray): Array to check.

        Returns:
            None.
        """
        assert np.all((arr >= 0) & (arr <= 255))


class TestPadToMultipleOf8(TensorHelpers):
    def _check_divisible_by_8(self, ten: torch.Tensor) -> None:
        """
        Checks if all dimensions of the tensor are divisible by 8.

        Args:
            ten (torch.Tensor): Tensor to check.

        Returns:
            None.
        """
        assert ten.shape[2] % 8 == 0
        assert ten.shape[3] % 8 == 0

    def _check_preserved_values(self, ten: torch.Tensor, pad_ten: torch.Tensor) -> None:
        """
        Checks if the original tensor values are preserved in the padded tensor.

        Args:
            ten (torch.Tensor): Original tensor.
            pad_ten (torch.Tensor): Padded tensor.

        Returns:
            None.
        """
        og_shape = ten.shape
        pad_ten_og_dims = pad_ten[
            : og_shape[0], : og_shape[1], : og_shape[2], : og_shape[3]
        ]
        assert torch.equal(ten, pad_ten_og_dims)

    def _check_idempotence(self, ten: torch.Tensor) -> None:
        """
        Checks if padding an already padded tensor does not change it.

        Args:
            ten (torch.Tensor): Padded tensor.

        Returns:
            None.
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

        Args:
            init_torch_tensor (torch.Tensor): Tensor representation of the tiff file.

        Returns:
            None.
        """
        ten = init_torch_tensor
        pad_ten = raft.pad_to_multiple_of_8(ten)

        self._check_divisible_by_8(pad_ten)
        self._check_preserved_values(ten, pad_ten)
        self._check_idempotence(pad_ten)

        B, C, H, W = ten.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8

        assert pad_ten.shape == (B, C, H+pad_h, W+pad_w)


class TestPreprocessTensor(TensorHelpers):
    def test_preprocess_tensor(
        self, init_tiff: tiff.Tiff, init_torch_tensor: torch.Tensor
    ) -> None:
        """
        Tests the preprocess_tensor function.

        Args:
            init_tiff (tuple): A tuple containing information about the TIFF file.
                - img (str): A TIFF instance.
                - f (int): Number of frames.
                - c (int): Number of channels.
                - h (int): Height.
                - w (int): Width.
            init_torch_tensor (torch.Tensor): Tensor representation of the tiff file.

        Returns:
            None.
        """
        tiff_file, info = init_tiff
        f, c, h, w = info

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

        Args:
            init_torch_tensor (torch.Tensor): Tensor representation of the tiff file.

        Returns:
            None.
        """
        ten = init_torch_tensor
        batch1, batch2 = raft.batch_frames(ten)

        assert batch1.shape == batch2.shape
        assert torch.equal(batch1[0], ten[0])
        assert torch.equal(batch1[-1], ten[-2])

        assert torch.equal(batch2[0], ten[1])
        assert torch.equal(batch2[-1], ten[-1])


class TestGetRAFTOpticalFlow:
    def _check_model_size_logic(
        self,
        ten: torch.Tensor,
    ) -> None:
        """
        Checks if the RAFT model size selection logic is correct.

        Args:
            ten (torch.Tensor): Tensor to check.

        Returns:
            None.
        """
        if ten.shape[0] >= 2:
            batch_1 = ten[:2]
            batch_2 = ten[:2]
        else:
            batch_1 = ten
            batch_2 = ten
        batches = (batch_1, batch_2)

        with (
            patch("src.cell_tracking.raft.raft_small") as mock_small,
            patch("src.cell_tracking.raft.raft_large") as mock_large,
        ):

            mock_model = Mock()
            mock_model.eval = Mock(return_value=None)
            mock_model.to = Mock(return_value=mock_model)
            mock_model.load_state_dict = Mock()
            flow_output = torch.zeros(
                batch_1.shape[0], 2, batch_1.shape[2], batch_1.shape[3]
            )
            mock_model.return_value = [flow_output]
            mock_small.return_value = mock_model
            mock_large.return_value = mock_model

            raft.get_raft_optical_flow(batches, model_size=raft.ModelSize.SMALL)
            mock_small.assert_called_once_with(progress=False)
            mock_large.assert_not_called()

        with (
            patch("src.cell_tracking.raft.raft_small") as mock_small,
            patch("src.cell_tracking.raft.raft_large") as mock_large,
        ):

            mock_model = Mock()
            mock_model.eval = Mock(return_value=None)
            mock_model.to = Mock(return_value=mock_model)
            mock_model.load_state_dict = Mock()
            flow_output = torch.zeros(
                batch_1.shape[0], 2, batch_1.shape[2], batch_1.shape[3]
            )
            mock_model.return_value = [flow_output]
            mock_small.return_value = mock_model
            mock_large.return_value = mock_model

            raft.get_raft_optical_flow(batches, model_size=raft.ModelSize.LARGE)
            mock_large.assert_called_once_with(progress=False)
            mock_small.assert_not_called()

    def _check_if_custom_weights_used(self, ten: torch.Tensor) -> None:
        """
        Checks if custom weights are used in the RAFT model.

        Args:
            ten (torch.Tensor): Tensor to check.

        Returns:
            None.
        """
        if ten.shape[0] >= 2:
            batch_1 = ten[:2]
            batch_2 = ten[:2]
        else:
            batch_1 = ten
            batch_2 = ten
        batches = (batch_1, batch_2)

        custom_weights = {"layer1.weight": torch.randn(10, 10)}

        with patch("src.cell_tracking.raft.raft_small") as mock_raft:
            mock_model = Mock()
            mock_model.eval = Mock(return_value=None)
            mock_model.to = Mock(return_value=mock_model)
            mock_model.load_state_dict = Mock()
            flow_output = torch.zeros(
                batch_1.shape[0], 2, batch_1.shape[2], batch_1.shape[3]
            )
            mock_model.return_value = [flow_output]
            mock_raft.return_value = mock_model

            raft.get_raft_optical_flow(batches, model_weights=custom_weights)
            mock_model.load_state_dict.assert_called_once_with(
                custom_weights, strict=False
            )

        with patch("src.cell_tracking.raft.raft_small") as mock_raft:
            mock_model = Mock()
            mock_model.eval = Mock(return_value=None)
            mock_model.to = Mock(return_value=mock_model)
            mock_model.load_state_dict = Mock()
            flow_output = torch.zeros(
                batch_1.shape[0], 2, batch_1.shape[2], batch_1.shape[3]
            )
            mock_model.return_value = [flow_output]
            mock_raft.return_value = mock_model

            raft.get_raft_optical_flow(batches, model_weights=None)
            mock_model.load_state_dict.assert_not_called()

    def _check_if_gpu_used(self, ten: torch.Tensor) -> None:
        """
        Checks if GPU is used in the RAFT model.

        Args:
            ten (torch.Tensor): Tensor to check.

        Returns:
            None.
        """
        if not torch.cuda.is_available():
            return None

        if ten.shape[0] >= 2:
            batch_1 = ten[:2]
            batch_2 = ten[:2]
        else:
            batch_1 = ten
            batch_2 = ten
        batches = (batch_1, batch_2)

        with (
            patch("src.cell_tracking.raft.raft_small") as mock_raft,
            patch("torch.cuda.is_available", return_value=True),
        ):

            mock_model = Mock()
            mock_model.eval = Mock(return_value=None)
            mock_model.to = Mock(return_value=mock_model)
            mock_model.load_state_dict = Mock()
            flow_output = torch.zeros(
                batch_1.shape[0], 2, batch_1.shape[2], batch_1.shape[3]
            )
            mock_model.return_value = [flow_output]
            mock_raft.return_value = mock_model

            raft.get_raft_optical_flow(batches, gpu_flag=True)

            call_args = mock_model.to.call_args_list[0][0][0]
            assert call_args.type == "cuda"

        with patch("src.cell_tracking.raft.raft_small") as mock_raft:
            mock_model = Mock()
            mock_model.eval = Mock(return_value=None)
            mock_model.to = Mock(return_value=mock_model)
            mock_model.load_state_dict = Mock()
            flow_output = torch.zeros(
                batch_1.shape[0], 2, batch_1.shape[2], batch_1.shape[3]
            )
            mock_model.return_value = [flow_output]
            mock_raft.return_value = mock_model

            raft.get_raft_optical_flow(batches, gpu_flag=True)
            call_args = mock_model.to.call_args_list[0][0][0]
            assert call_args.type == "cuda"

        with (
            patch("src.cell_tracking.raft.raft_small") as mock_raft,
            patch("torch.cuda.is_available", return_value=False),
        ):

            mock_model = Mock()
            mock_model.eval = Mock(return_value=None)
            mock_model.to = Mock(return_value=mock_model)
            mock_model.load_state_dict = Mock()
            flow_output = torch.zeros(
                batch_1.shape[0], 2, batch_1.shape[2], batch_1.shape[3]
            )
            mock_model.return_value = [flow_output]
            mock_raft.return_value = mock_model

    def test_get_raft_optical_flow(self, init_torch_tensor: torch.Tensor) -> None:
        """
        Tests the get_raft_optical_flow function.

        Args:
            init_torch_tensor (torch.Tensor): Tensor representation of the tiff file.

        Returns:
            None.
        """
        ten = init_torch_tensor

        if ten.shape[0] >= 2:
            batch_1 = ten[:2]
            batch_2 = ten[:2]
        else:
            batch_1 = ten
            batch_2 = ten

        self._check_model_size_logic(ten)
        self._check_if_custom_weights_used(ten)
        self._check_if_gpu_used(ten)

        with patch("src.cell_tracking.raft.raft_small") as mock_raft:
            expected_flow = torch.randn(
                batch_1.shape[0], 2, batch_1.shape[2], batch_1.shape[3]
            )
            mock_model = Mock()
            mock_model.eval = Mock(return_value=None)
            mock_model.to = Mock(return_value=mock_model)
            mock_model.load_state_dict = Mock()
            mock_model.return_value = [expected_flow]
            mock_raft.return_value = mock_model

            batches = (batch_1, batch_2)

            if torch.cuda.is_available():
                result = raft.get_raft_optical_flow(batches, gpu_flag=True)
            else:
                result = raft.get_raft_optical_flow(batches, gpu_flag=False)

            assert result.shape == (
                batch_1.shape[0],
                2,
                batch_1.shape[2],
                batch_1.shape[3],
            )

            assert mock_model.call_count == 1

            mock_model.eval.assert_called_once()

            assert result.device.type == "cpu"

            assert mock_model.to.call_count >= 1


class TestMakeRAFTOutputArray(NdarrayHelpers):
    def test_make_raft_output_array(self) -> None:
        """
        Tests the make_raft_output_array function using synthetic flow data.

        Args:
            None.

        Returns:
            None.
        """
        flow = torch.randn(5, 2, 128, 256)

        arr = raft.make_raft_output_array(flow)

        expected_shape = (5, 128, 256, 2)
        self._check_if_float32_np(arr)
        self._check_shape_np(arr, expected_shape)

        assert isinstance(arr, np.ndarray)

    def test_make_raft_output_array_single_frame(self) -> None:
        """
        Tests with a single frame.

        Args:
            None.

        Returns:
            None.
        """
        flow = torch.randn(1, 2, 64, 64)
        arr = raft.make_raft_output_array(flow)

        assert arr.shape == (1, 64, 64, 2)
        assert isinstance(arr, np.ndarray)

    def test_make_raft_output_array_dimension_order(self) -> None:
        """
        Tests that dimensions are correctly transposed from (f, 2, h, w) to (f, h, w, 2).

        Args:
            None.

        Returns:
            None.
        """
        flow = torch.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4).float()
        arr = raft.make_raft_output_array(flow)

        assert arr.shape == (2, 3, 4, 2)
        assert arr[0, 0, 0, 0] == flow[0, 0, 0, 0].item()
        assert arr[0, 0, 0, 1] == flow[0, 1, 0, 0].item()


class TestCalcOpticalFlowRAFT:
    def test_calcOpticalFlowRAFT(self, init_tiff: tiff.Tiff) -> None:
        """
        Tests the calcOpticalFlowRAFT function.

        Args:
            init_tiff (tuple): A tuple containing information about the TIFF file.
                - img (str): A TIFF instance.
                - f (int): Number of frames.
                - c (int): Number of channels.
                - h (int): Height.
                - w (int): Width.

        Returns: 
            None.
        """
        tiff_file, info = init_tiff
        f, c, h, w = info

        with patch("src.cell_tracking.raft.raft_small") as mock_raft:
            mock_model = type(
                "MockModel",
                (),
                {
                    "eval": lambda self: None,
                    "to": lambda self, device: self,
                    "load_state_dict": lambda self, weights, strict: None,
                    "__call__": lambda self, b1, b2: [
                        torch.zeros(b1.shape[0], 2, b1.shape[2], b1.shape[3])
                    ],
                },
            )()
            mock_raft.return_value = mock_model

            """
            , \
            patch("src.cell_tracking.raft.preprocess_tensor") as mock_preprocess_tensor, \
            patch("src.cell_tracking.raft.batch_frames") as mock_batch_frames, \
            patch("src.cell_tracking.raft.get_raft_optical_flow") as mock_get_raft_optical_flow, \
            patch("src.cell_tracking.raft.make_raft_output_array") as mock_make_raft_output_array
            """

            #make return values/side effects

            result = raft.calcOpticalFlowRAFT(tiff_file)

            #assert args, kwargs for four mocked funcs
            #assert called for four mocked funcs

            assert isinstance(result, np.ndarray)
            assert result.ndim == 4
            assert result.shape[3] == 2

            expected_frames = tiff_file.arr.shape[0] - 1
            assert result.shape[0] == expected_frames

            assert result.shape == (f, h, w, 2)

    def test_calcOpticalFlowRAFT_with_custom_params(self, init_tiff: tiff.Tiff) -> None:
        """
        Tests calcOpticalFlowRAFT with custom parameters.

        Args:
            init_tiff (tuple): A tuple containing information about the TIFF file.
                - img (str): A TIFF instance.
                - f (int): Number of frames.
                - c (int): Number of channels.
                - h (int): Height.
                - w (int): Width.

        Returns:
            None.
        """
        tiff_file, info = init_tiff
        f, c, h, w = info

        custom_weights = {"layer1.weight": torch.randn(10, 10)}

        with patch("src.cell_tracking.raft.raft_large") as mock_raft:
            mock_model = type(
                "MockModel",
                (),
                {
                    "eval": lambda self: None,
                    "to": lambda self, device: self,
                    "load_state_dict": lambda self, weights, strict: None,
                    "__call__": lambda self, b1, b2: [
                        torch.zeros(b1.shape[0], 2, b1.shape[2], b1.shape[3])
                    ],
                },
            )()
            mock_raft.return_value = mock_model

            result = raft.calcOpticalFlowRAFT(
                tiff_file,
                model_size=raft.ModelSize.LARGE,
                model_weights=custom_weights,
                gpu_flag=False,
            )

            assert isinstance(result, np.ndarray)
            assert result.shape[3] == 2

            mock_raft.assert_called_once_with(progress=False)