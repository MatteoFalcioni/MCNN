import torch
from sklearn.metrics import precision_score
from scripts.point_cloud_to_image import generate_multiscale_grids
import numpy as np


def inference(model, data_array, window_sizes, grid_resolution, channels, device, true_labels=None):
    """
    Perform inference with the MCNN model, generating grids from an array of point cloud
    points before classification. Also calculates precision if true labels are provided.

    Args:
    - model (nn.Module): The trained MCNN model.
    - data_array (np.ndarray): The array of points from the point cloud (e.g., [x, y, z, ...]).
    - window_sizes (list): List of window sizes for grid generation.
    - grid_resolution (int): Resolution of the grid (e.g., 128x128).
    - channels (int): Number of channels in the grid.
    - device (torch.device): The device (CPU or GPU) to perform inference on.
    - true_labels (torch.Tensor or np.ndarray): The true class labels for calculating precision (optional).

    Returns:
    - predicted_labels (torch.Tensor): Predicted class labels.
    - precision (float): Precision score (only if true_labels are provided).
    """
    # Generate multiscale grids for the point cloud
    grids_dict = generate_multiscale_grids(
        data_array=data_array,
        window_sizes=window_sizes,
        grid_resolution=grid_resolution,
        channels=channels,
        save=False  # We don't need to save grids during inference
    )

    # Extract the small, medium, and large grids from the generated dictionary
    small_grids = torch.tensor(np.array(grids_dict['small']['grids']), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    medium_grids = torch.tensor(np.array(grids_dict['medium']['grids']), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    large_grids = torch.tensor(np.array(grids_dict['large']['grids']), dtype=torch.float32).permute(0, 3, 1, 2).to(device)

    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():  # No need to track gradients during inference
        # Forward pass through the model
        outputs = model(small_grids, medium_grids, large_grids)

        # Get predicted class (highest probability) for each sample
        _, predicted_labels = torch.max(outputs, dim=1)

    precision = None
    if true_labels is not None:
        if isinstance(true_labels, np.ndarray):
            true_labels = torch.tensor(true_labels)
        precision = precision_score(true_labels.cpu(), predicted_labels.cpu(), average='macro', zero_division=0)

    return predicted_labels, precision
