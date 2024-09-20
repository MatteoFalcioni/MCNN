import torch
import numpy as np


def select_device():
    """
    Selects the best available device for PyTorch: CUDA > DirectML > CPU.

    Returns:
    - torch.device: The selected device (CUDA, DirectML, or CPU).
    """
    try:
        if torch.cuda.is_available():
            print("Using CUDA device.")
            return torch.device('cuda')
        elif torch.directml.is_available():
            print("CUDA not available. Using DirectML device.")
            return torch.device('dml')
        else:
            print("CUDA and DirectML not available. Using CPU.")
            return torch.device('cpu')
    except Exception as e:
        print(f"Error selecting device: {e}")
        return torch.device('cpu')


def move_to_device(array, device):
    """
    Moves a numpy array or torch tensor to the specified device (GPU/CPU).
    Only converts NumPy arrays to tensors if a GPU is available.

    Args:
    - array: A numpy array or torch tensor to move.
    - device: The device to move the tensor/array to (CPU or GPU).

    Returns:
    - tensor/array: The object moved to the correct device (as a tensor if on GPU).
    """
    if isinstance(array, np.ndarray):
        if device.type == 'cuda':
            # Convert to tensor and move to GPU
            return torch.tensor(array).to(device)
        else:
            # If we're on CPU, just return the numpy array (no conversion)
            return array
    elif isinstance(array, torch.Tensor):
        # If it's already a tensor, move it to the correct device
        return array.to(device)
    return array  # If it's something else, return as is

