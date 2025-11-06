import torch
import torchvision.transforms.functional as F
import torch.nn.functional as F
import torchvision.models.optical_flow as torch_of
import numpy as np
import tiffclass as tiff
import enum


class ModelSize(enum.IntEnum):
    SMALL = 1
    LARGE = 2


def pad_to_multiple_of_8(ten : torch.Tensor) -> torch.Tensor:
    """
    Pads tensor to make the height and width divisible by 8.
    This is required for the RAFT model to work.

    Args:
        ten: A 4D PyTorch tensor.

    Returns:
        torch.Tensor: ten, with a padded height and width.
    """
    _, _, H, W = ten.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8

    return F.pad(ten, (0, pad_w, 0, pad_h), mode='replicate')



def make_tiff_into_tensor(tiff_file: tiff.Tiff) -> torch.Tensor:
    """
    Takes an instance of a tiff.Tiff, extracts the img array and
    casts it as a tensor.Torch.

    Args:
        tiff_file: An instance of a tiff.Tiff.

    Returns:
        torch.Tensor: a torch.Tensor version of tiff_file.img.
    """
    arr = tiff_file.arr
    arr = arr.astype('float32') / 65535.0

    ten = torch.from_numpy(arr)
    ten = ten.unsqueeze(1)
    ten = ten.repeat(1, 3, 1, 1)  # [T, 3, H, W]
    ten = pad_to_multiple_of_8(ten)

    return ten


def preprocess_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Applies padding to the tensor. Additionally, copies the
    phase contrast channel and stacks it on itself twice.
    Creates a copy of t that is ready to use in the
    RAFT model.

    Args:
        t: A torch.Tensor representation of a tiff video.

    Returns:
        torch.Tensor: A preprocessed version of t.
    """
    raise NotImplementedError


def batch_frames(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batches frames together for use in RAFT model.

    Args:
        t: A torch.Tensor representation of a tiff video.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of torch.Tensors.
                The first tensor is t[0:len(t) - 1]. The second is
                t[1:len(t)].
    """
    raise NotImplementedError


def calculate_raft_optical_flow(
    batches: tuple[torch.Tensor, torch.Tensor],
    model_size: int = ModelSize.SMALL,
    gpu_flag: bool = False,
) -> torch.Tensor:
    """
    Determines the optical_flow of a tiff file using the
    RAFT algorithm.

    Args:
        batches: The two torch.Tensors that act as inputs for the
            raft models.
        model_size: If the small RAFT model or large RAFT model
            should be use. The default is small.
        gpu_flag: If the GPU should be used if available. Marking
            this flag as False will use CPU no matter what. The
            default is False.

    Returns:
        torch.Tensor: Optical flow of the tiff file represented by
            the batches.
    """
    raise NotImplementedError


def make_raft_output_array(flow: torch.Tensor) -> np.ndarray:
    """
    Casts the optical flow of a tiff file, flow as an np.array.

    Args:
        flow: The torch.Tensor representation of optical flow.

    Returns:
        np.ndarray: The same representation, but as an np.ndarray.
    """
    raise NotImplementedError
