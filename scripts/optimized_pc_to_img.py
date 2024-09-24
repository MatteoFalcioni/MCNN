import torch
from scipy.spatial import KDTree
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from utils.point_cloud_data_utils import remap_labels


def gpu_create_feature_grid(center_point, window_size, grid_resolution=128, channels=3, device=None):
    """
    Creates a grid around the center point and initializes cells to store feature values on the GPU or CPU.

    Args:
    - center_point (tuple): The (x, y, z) coordinates of the center point of the grid.
    - window_size (float): The size of the square window around the center point (in meters).
    - grid_resolution (int): The number of cells in one dimension of the grid (e.g., 128 for a 128x128 grid). Default is 128.
    - channels (int): The number of channels in the resulting image. Default is 3.
    - device (torch.device): The device to create tensors on (CPU or GPU).

    Returns:
    - grid (torch.Tensor): A 2D grid initialized to zeros, which will store feature values.
    - cell_size (float): The size of each cell in meters.
    - x_coords (torch.Tensor): Tensor of x coordinates for the centers of the grid cells.
    - y_coords (torch.Tensor): Tensor of y coordinates for the centers of the grid cells.
    - z_coords (torch.Tensor): Tensor of z coordinates for the centers of the grid cells.
    """
    # Calculate the size of each cell in meters
    cell_size = window_size / grid_resolution

    # Initialize the grid to zeros; each cell will eventually hold feature values (using torch)
    grid = torch.zeros((grid_resolution, grid_resolution, channels), device=device)

    # Generate cell coordinates for the grid based on the center point
    i_indices = torch.arange(grid_resolution, device=device)
    j_indices = torch.arange(grid_resolution, device=device)

    half_resolution_plus_half = (grid_resolution / 2) + 0.5

    # following x_k = x_pk - (64.5 - j) * w
    x_coords = center_point[0] - (half_resolution_plus_half - j_indices) * cell_size
    y_coords = center_point[1] - (half_resolution_plus_half - i_indices) * cell_size
    z_coords = torch.full((grid_resolution, grid_resolution), center_point[2], device=device)  # Z coordinate is constant for all cells

    return grid, cell_size, x_coords, y_coords, z_coords


def gpu_assign_features_to_grid(data_tensor, grid, x_coords, y_coords, channels=3, device=None):
    """
    Assign features from the nearest point to each cell in the grid using PyTorch.

    Args:
    - data_tensor (torch.Tensor): Tensor where each row represents a point with its x, y, z coordinates and features.
    - grid (torch.Tensor): A 2D grid initialized to zeros, which will store feature values.
    - x_coords (torch.Tensor): Tensor of x coordinates for the centers of the grid cells.
    - y_coords (torch.Tensor): Tensor of y coordinates for the centers of the grid cells.
    - channels (int): Number of feature channels to assign to each grid cell (default is 3 for RGB).
    - device (torch.device): The device to run this on (CPU or GPU).

    Returns:
    - grid (torch.Tensor): Grid populated with feature values.
    """
    # Extract point coordinates (x, y) and features for KDTree equivalent with PyTorch
    points = data_tensor[:, :2]  # Assuming x, y are the first two columns
    features = data_tensor[:, 3:3 + channels]  # Assuming features start from the 4th column (index 3)

    # Stack the coordinates into a 2D grid for comparison
    grid_coords = torch.stack(torch.meshgrid(x_coords, y_coords), dim=-1).view(-1, 2)  # Flattened grid coordinates

    # Compute distances between each point in the point cloud and each grid cell using torch.cdist
    dists = torch.cdist(grid_coords.to(device), points.to(device))  # Compute pairwise distances

    # Find the index of the closest point for each grid cell
    closest_points_idx = torch.argmin(dists, dim=1)

    # Reshape the closest points into the grid shape and assign features
    for channel in range(channels):
        grid[:, :, channel] = features[closest_points_idx, channel].view(grid.shape[0], grid.shape[1])

    return grid


def batch_process(data_loader, window_sizes, grid_resolution, channels, device, save_dir=None, save=False):
    """
    Batch processing for GPU-accelerated grid generation.
    Process the entire batch of data and generate grids with features in parallel.

    Arguments:
    data_loader -- DataLoader containing batches of point clouds, features, and labels.
    window_sizes -- List of tuples containing the label and window size (e.g., [('small', 10), ('medium', 20)]).
    grid_resolution -- Resolution of the grid to create (e.g., 128x128).
    channels -- Number of feature channels.
    device -- The device (GPU or CPU) to run this on.
    save_dir -- Directory to save the grids (if required).
    save -- Boolean flag to control whether to save grids or not.

    Returns:
    labeled_grids_dict -- A dictionary with grids and class labels for each scale.
    """
    # Initialize a dictionary to store grids and class labels for each scale
    labeled_grids_dict = {scale_label: {'grids': [], 'class_labels': []} for scale_label, _ in window_sizes}

    # Iterate over batches from the data loader
    for batch_idx, (batch_data, batch_features, batch_labels) in enumerate(data_loader):
        print(f"Processing batch {batch_idx + 1}/{len(data_loader)} with {len(batch_data)} points")

        # Move entire batch to the device
        batch_data = batch_data.to(device)
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        for size_label, window_size in window_sizes:
            print(f"Generating {size_label} grid for batch {batch_idx} with window size {window_size}...")

            # Create a batch of grids
            grids, _, x_coords, y_coords = gpu_create_feature_grid(batch_data, window_size, grid_resolution,
                                                                   channels, device)

            # Assign features for the entire batch to the grids
            grids = gpu_assign_features_to_grid(batch_data, batch_features, grids, x_coords, y_coords, channels, device)

            # Store the grids and labels in the dictionary
            labeled_grids_dict[size_label]['grids'].append(grids)
            labeled_grids_dict[size_label]['class_labels'].append(batch_labels)

            # Optionally save the grids to the disk
            if save and save_dir is not None:
                for i, grid in enumerate(grids):
                    grid_with_features_np = grid.cpu().numpy()
                    scale_dir = os.path.join(save_dir, size_label)
                    os.makedirs(scale_dir, exist_ok=True)
                    grid_filename = os.path.join(scale_dir,
                                                 f"grid_{batch_idx}_{i}_{size_label}_class_{int(batch_labels[i])}.npy")
                    np.save(grid_filename, grid_with_features_np)
                    print(f"Saved {size_label} grid for batch {batch_idx}, point {i} to {grid_filename}")

    return labeled_grids_dict


def gpu_generate_multiscale_grids(data_array, window_sizes, grid_resolution, channels, device, save_dir=None,
                                  save=False, batch_size=50, num_workers=4):
    """
    Optimized multiscale grid generation using Torch tensors with parallel batching.

    Arguments:
    data_array -- Numpy array of data with features and labels.
    window_sizes -- List of window sizes to generate grids for different scales.
    grid_resolution -- Resolution of the grid to create (e.g., 128x128).
    channels -- Number of feature channels.
    device -- The device (GPU or CPU) to run this on.
    save_dir -- Directory to save the generated grids (if required).
    save -- Boolean flag to control whether to save grids or not.
    batch_size -- Batch size for processing.
    num_workers -- Number of workers for parallel data loading.

    Returns:
    labeled_grids_dict -- Dictionary containing multiscale grids and corresponding labels.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Remap labels in the data array (if necessary)
    data_array, _ = remap_labels(data_array)

    # Include the features along with the coordinates (x, y, z) in the dataset
    dataset = TensorDataset(torch.tensor(data_array[:, :3]), torch.tensor(data_array[:, 3:3 + channels]),
                            torch.tensor(data_array[:, -1]))  # Including class labels as well
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Process the data in batches
    return batch_process(data_loader, window_sizes, grid_resolution, channels, device, save_dir, save)


# in the batch function we select a batch (batch_index). then we pass it to batch feature assignment,
# but like this we shouldnt expect the tensors
# to be 3d, because the batch dim is already selected! then either we don loop inside the batch_idx,
# or we adjust batch feature assignment to work wi

# but probably this was a mistake as well. need to go deeper in the process

