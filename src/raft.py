import enum
import torch
import numpy as np
import tiffclass as tiff
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, raft_small


class ModelSize(enum.IntEnum):
    SMALL = 1
    LARGE = 2


def pad_to_multiple_of_8(ten: torch.Tensor) -> torch.Tensor:
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

    return F.pad(ten, (0, pad_w, 0, pad_h), mode="replicate")


def preprocess_tensor(tiff_file: tiff.Tiff, **kwargs) -> torch.Tensor:
    """
    Loads and preprocesses a TIFF stack for RAFT. Converts to
    float32, normalizes to [0, 1], repeats channels to RGB,
    and pads height/width to multiples of 8.

    Args:
        tiff_file: The tiff file representing the video to be processed.
            It holds an array of shape [frames, channels, height, width].

    Returns:
        torch.Tensor: A preprocessed representation of the tiff_file; ready
            to be used in the RAFT model.
    """
    arr = tiff_file.arr
    arr = arr[:, 0, ...]
    arr = tiff_file.preprocess_stack(arr, **kwargs)
    arr = arr.astype("float32") / np.iinfo(arr.dtype).max

    ten = (
        torch.from_numpy(arr).unsqueeze(1).repeat(1, 3, 1, 1)
    )  # [frames, 3, height, width]
    ten = pad_to_multiple_of_8(ten)

    return ten


def batch_frames(ten: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batches frames together for use in RAFT model.

    Args:
        t: A torch.Tensor representation of a tiff video.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of torch.Tensors.
                The first tensor is t[0:len(t) - 1]. The second is
                t[1:len(t)].
    """
    return ten[:-1], ten[1:]


def get_raft_optical_flow(
    batches: tuple[torch.Tensor, torch.Tensor],
    model_size: ModelSize = ModelSize.SMALL,
    model_weights: dict | None = None,
    device_flag: bool = False,
) -> torch.Tensor:
    """
    Computes optical flow between frame pairs using the RAFT model.

    Args:
        batches: Tuple of two tensors (batch_1, batch_2),
                 each of shape [f, 3, h, w].
        model_size: Which RAFT variant to use (SMALL or LARGE).
        model_weights: A loaded state_dict for the model, or None
                       to use default pretrained weights.
        device_flag: If True, use CUDA when available; else CPU.

    Returns:
        torch.Tensor: Optical flow tensor [f, 2, h, w].
    """
    device = torch.device(
        "cuda" if device_flag and torch.cuda.is_available() else "cpu"
    )

    if model_size == ModelSize.SMALL:
        model = raft_small(progress=False).to(device)
    else:
        model = raft_large(progress=False).to(device)

    if model_weights is not None:
        model.load_state_dict(model_weights, strict=False)

    model.eval()
    batch_1, batch_2 = batches

    with torch.no_grad():
        list_of_flows = model(batch_1.to(device), batch_2.to(device))
        flow = list_of_flows[-1]

    return flow.cpu()


def make_raft_output_array(flow: torch.Tensor) -> np.ndarray:
    """
    Casts the optical flow of a tiff file, flow as an np.array.

    Args:
        flow: The torch.Tensor representation of optical flow.
            The shape is [f, 2, h, w]

    Returns:
        np.ndarray: The same representation, but as an np.ndarray.
            The shape is [f, h, w, 2] to match the outputs of the
            other optical flow models.
    """
    ten = torch.permute(flow, (0, 2, 3, 1))
    arr = torch.Tensor.numpy(ten)
    return arr
