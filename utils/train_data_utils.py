import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_synthetic_data(num_samples=1000):
    """
    Generates synthetic data for model training. This function creates random 3-channel (e.g., RGB) images
    of size 128x128 and random labels for classification tasks.

    Args:
    - num_samples (int): The number of synthetic data samples to generate.

    Returns:
    - X (torch.Tensor): A tensor of shape (num_samples, 3, 128, 128) representing the input images.
    - y (torch.Tensor): A tensor of shape (num_samples,) representing the random labels (0-8 for 9 classes).
    """
    X = torch.randn(num_samples, 3, 128, 128)  # Input images
    y = torch.randint(0, 9, (num_samples,))  # Labels (0-8)
    return X, y


def prepare_dataloader(batch_size, num_samples=1000):
    """
    Prepares a DataLoader for training using synthetic data. This function generates synthetic data and
    then wraps it in a DataLoader object for mini-batch training.

    Args:
    - batch_size (int): The size of each mini-batch.
    - num_samples (int): The number of synthetic data samples to generate.

    Returns:
    - dataloader (DataLoader): A DataLoader object for the synthetic data.
    """
    X, y = generate_synthetic_data(num_samples)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


