import torch
import torchvision.transforms.functional as F
import torchvision.models.optical_flow as torch_of
import numpy as np
import tiffclass as tiff
import enum

class ModelSize(enum.IntEnum):
    SMALL = 1
    LARGE = 2

# ModelSize = enum.Enum('ModelSize', [('Small', 1), ('Large', 2)]) 

def make_tiff_into_tensor(tiff_file : tiff.Tiff) -> torch.Tensor :
    """
    Takes an instance of a tiff.Tiff, extracts the img array and
    casts it as a tensor.Torch.

    Args:
        tiff_file: An instance of a tiff.Tiff.

    Returns:
        torch.Tensor: a torch.Tensor version of tiff_file.img.
    """
    raise NotImplementedError

def preprocess_tensor(t : torch.Tensor) -> torch.Tensor:
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

def batch_frames(t : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

def calculate_raft_optical_flow(batches : tuple[torch.Tensor, torch.Tensor], 
                                model_size : int = ModelSize.SMALL,
                                gpu_flag : bool = False)  -> torch.Tensor:
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

def make_raft_output_array(flow : torch.Tensor) -> np.ndarray:
    """
    Casts the optical flow of a tiff file, flow as an np.array.

    Args:
        flow: The torch.Tensor representation of optical flow.

    Returns:
        np.ndarray: The same representation, but as an np.ndarray.
    """
    raise NotImplementedError

