import torch
from scipy.spatial import KDTree
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from utils.point_cloud_data_utils import remap_labels


def gpu_create_feature_grid(center_points, window_size, grid_resolution=128, channels=3, device=None):
    """
    Creates a batch of grids around the center points and initializes cells to store feature values.

    Args:
    - center_points (torch.Tensor): A tensor of shape [batch_size, 3] containing (x, y, z) coordinates of the center points.
    - window_size (float): The size of the square window around each center point (in meters).
    - grid_resolution (int): The number of cells in one dimension of the grid (e.g., 128 for a 128x128 grid).
    - channels (int): The number of channels in the resulting image. Default is 3 for RGB.
    - device (torch.device): The device (CPU or GPU) where tensors will be created.

    Returns:
    - grids (torch.Tensor): A tensor of shape [batch_size, channels, grid_resolution, grid_resolution].
    - cell_size (float): The size of each cell in meters.
    - x_coords (torch.Tensor): A tensor of shape [batch_size, grid_resolution] for x coordinates of grid cells.
    - y_coords (torch.Tensor): A tensor of shape [batch_size, grid_resolution] for y coordinates of grid cells.
    """

    center_points = center_points.to(device)    # additional check to avoid cpu usage
    batch_size = center_points.shape[0]  # Number of points in the batch

    # Calculate the size of each cell in meters
    cell_size = window_size / grid_resolution

    # Initialize the grids with zeros; one grid for each point in the batch
    grids = torch.zeros((batch_size, channels, grid_resolution, grid_resolution), device=device)

    # Generate grid coordinates for each point in the batch
    half_resolution_plus_half = (grid_resolution / 2) + 0.5

    # Create x and y coordinate grids for each point in the batch
    x_coords = center_points[:, 0].unsqueeze(1) - (half_resolution_plus_half - torch.arange(grid_resolution, device=device).view(1, -1)) * cell_size
    y_coords = center_points[:, 1].unsqueeze(1) - (half_resolution_plus_half - torch.arange(grid_resolution, device=device).view(1, -1)) * cell_size

    return grids, cell_size, x_coords, y_coords


def gpu_assign_features_to_grid(batch_data, batch_features, grids, x_coords, y_coords, channels=3, device=None):
    """
    Assign features from the nearest point to each cell in the grid for a batch of points.

    Args:
    - batch_data (torch.Tensor): A tensor of shape [batch_size, num_points, 2] representing the (x, y) coordinates of point cloud.
    - batch_features (torch.Tensor): A tensor of shape [batch_size, num_points, num_features] representing the features.
    - grids (torch.Tensor): A tensor of shape [batch_size, channels, grid_resolution, grid_resolution].
    - x_coords (torch.Tensor): A tensor of shape [batch_size, grid_resolution] containing x coordinates for each grid cell.
    - y_coords (torch.Tensor): A tensor of shape [batch_size, grid_resolution] containing y coordinates for each grid cell.
    - channels (int): Number of feature channels in the grid (default is 3 for RGB).
    - device (torch.device): The device (CPU or GPU) to run this on.

    Returns:
    - grids (torch.Tensor): The updated grids with features assigned based on nearest points for the entire batch.
    """
    batch_size = batch_data.shape[0]  # Number of points in the batch
    num_available_features = batch_features.shape[2]  # How many features are available

    # Ensure we only extract up to 'channels' features
    batch_features = batch_features[:, :, :min(channels, num_available_features)]

    for i in range(batch_size):
        # Flatten grid coordinates for the i-th batch
        grid_coords = torch.stack([x_coords[i].reshape(-1), y_coords[i].reshape(-1)], dim=1).to(
            device)  # [grid_resolution^2, 2]

        # Use torch.cdist to compute distances between grid cells and points in the batch
        points = batch_data[i].to(device)  # Points (x, y) for the i-th batch
        dists = torch.cdist(grid_coords, points)  # Compute pairwise distances

        # Find the nearest points for each grid cell
        closest_points_idx = torch.argmin(dists, dim=1)

        # Assign features to the grid for the i-th batch
        for channel in range(channels):
            grids[i, channel, :, :] = batch_features[i, closest_points_idx, channel]

    return grids


def prepare_grids_dataloader(data_array, channels, batch_size, num_workers, device):
    """
    Prepares the DataLoader for batching the point cloud data.

    Args:
    - data_array (np.ndarray): The dataset containing (x, y, z) coordinates, features, and class labels.
    - channels (int): Number of feature channels in the data.
    - batch_size (int): The number of points to process in each batch.
    - num_workers (int): Number of workers for data loading.
    - device (torch.device): The device to move data to (CPU or GPU).

    Returns:
    - data_loader (DataLoader): A DataLoader that batches the dataset.
    """
    dataset = TensorDataset(
        torch.tensor(data_array[:, :3]),  # Center points (x, y, z)
        torch.tensor(data_array[:, 3:3 + channels]),  # Features
        torch.tensor(data_array[:, -1])  # Class labels
    )

    # Set num_workers > 0 to enable multiprocessing
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return data_loader

