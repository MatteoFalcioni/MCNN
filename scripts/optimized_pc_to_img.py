import torch
from scipy.spatial import KDTree
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from utils.point_cloud_data_utils import remap_labels


def gpu_create_feature_grid(center_points, window_size, grid_resolution=128, channels=3, device=None):
    """
    Optimized to work with a batch of center points for GPU acceleration.
    Batch version of the grid generation function.

    Arguments:
    center_points -- A tensor of shape [batch_size, 3] (x, y, z) for each point in the batch.
    window_size -- Size of the window to create grids.
    grid_resolution -- Resolution of the grid (default is 128x128).
    channels -- Number of feature channels in the grid.
    device -- The device (GPU or CPU) to run this on.

    Returns:
    grids -- A tensor of shape [batch_size, channels, grid_resolution, grid_resolution]
    x_coords -- A tensor of shape [batch_size, grid_resolution] representing x coordinates for each grid cell.
    y_coords -- A tensor of shape [batch_size, grid_resolution] representing y coordinates for each grid cell.
    """
    batch_size = center_points.shape[0]

    # Calculate the size of each cell in meters
    cell_size = window_size / grid_resolution

    # Initialize the grids with zeros using Torch tensors (one grid per batch item)
    grids = torch.zeros((batch_size, channels, grid_resolution, grid_resolution), device=device)

    # Calculate half the resolution plus 0.5 to center the grid coordinates
    half_resolution_plus_half = (grid_resolution / 2) + 0.5

    # Create x and y coordinate grids for each point in the batch
    x_coords = center_points[:, 0].unsqueeze(1) - (
                half_resolution_plus_half - torch.arange(grid_resolution, device=device).view(1, -1)) * cell_size
    y_coords = center_points[:, 1].unsqueeze(1) - (
                half_resolution_plus_half - torch.arange(grid_resolution, device=device).view(1, -1)) * cell_size

    return grids, cell_size, x_coords, y_coords


def gpu_assign_features_to_grid(batch_data, batch_features, batch_grids, x_coords, y_coords, channels=3, device=None):
    """
    Optimized feature assignment using Torch tensors and KDTree batching.
    Now accepts 'features' and 'grids' in batches.

    Arguments:
    batch_data -- A tensor of shape [batch_size, num_points, 2] representing the point cloud coordinates.
    batch_features -- A tensor of shape [batch_size, num_points, num_features] representing the features.
    grids -- A tensor of shape [batch_size, channels, grid_resolution, grid_resolution].
    x_coords, y_coords -- Grids of coordinates for each grid cell.
    channels -- Number of feature channels in the grid.
    device -- The device (GPU or CPU) to run this on.

    Returns:
    grids -- The updated grids with features assigned based on KDTree.
    """

    points = batch_data  # x, y coordinates in Torch
    batch_size = points.shape[0]

    print(f"Batch Features shape: {batch_features.shape}")
    num_available_features = batch_features.shape[2]  # Compute how many features are available
    print(f'Number of available features: {num_available_features}')

    # Ensure we only extract up to 'channels' features
    batch_features = batch_features[:, :, :min(channels, num_available_features)]

    print(f"Data points shape: {points.shape}")
    print(f"Features shape: {batch_features.shape}")

    # Use KDTree to assign features for the entire batch without looping
    points_cpu = points.cpu().numpy()  # Convert batch to CPU for KDTree
    tree = KDTree(points_cpu.reshape(-1, 2))  # Flatten batch for KDTree query (all points together)

    # Flatten grid coordinates for KDTree query
    flat_coords = torch.stack([x_coords, y_coords], dim=2).reshape(batch_size, -1, 2).cpu().numpy()

    # Query KDTree for the closest points for all batches
    _, idxs = tree.query(flat_coords.reshape(-1, 2))

    # Reshape idxs back to batch size for proper assignment
    idxs = idxs.reshape(batch_size, -1)

    if len(idxs) > 0:
        # Assign features for the entire batch using KDTree indices
        batch_grids = batch_features[:, idxs].transpose(1, 2)
    else:
        print("Warning: No features found.")

    return batch_grids


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
    for (batch_data, batch_features, batch_labels) in enumerate(data_loader):
        # print(f"Processing batch {batch_idx + 1}/{len(data_loader)} with {len(batch_data)} points")

        # Move entire batch to the device
        batch_data = batch_data.to(device)
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        for size_label, window_size in window_sizes:
            # print(f"Generating {size_label} grid for batch {batch_idx} with window size {window_size}...")

            # Create a batch of grids
            grids, _, x_coords, y_coords = gpu_create_feature_grid(batch_data, window_size, grid_resolution,
                                                                   channels, device)

            # Assign features for the entire batch to the grids
            grids = gpu_assign_features_to_grid(batch_data, batch_features, grids, x_coords, y_coords, channels, device)

            # Store the grids and labels in the dictionary
            labeled_grids_dict[size_label]['grids'].append(grids)
            labeled_grids_dict[size_label]['class_labels'].append(batch_labels)

            """# Optionally save the grids to the disk
            if save and save_dir is not None:
                for i, grid in enumerate(grids):
                    grid_with_features_np = grid.cpu().numpy()
                    scale_dir = os.path.join(save_dir, size_label)
                    os.makedirs(scale_dir, exist_ok=True)
                    grid_filename = os.path.join(scale_dir,
                                                 f"grid_{batch_idx}_{i}_{size_label}_class_{int(batch_labels[i])}.npy")
                    np.save(grid_filename, grid_with_features_np)
                    print(f"Saved {size_label} grid for batch {batch_idx}, point {i} to {grid_filename}")"""

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
# or we adjust batch feature assignment to work with a selected batch

