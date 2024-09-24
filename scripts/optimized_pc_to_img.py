import torch
from scipy.spatial import KDTree
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import KDTree


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


def gpu_assign_features_to_grid(batch_data, batch_features, grids, x_coords, y_coords, full_data, channels=3, device=None):
    """
    Assign features from the nearest point to each cell in the grid for a batch of points using KDTree.

    Args:
    - batch_data (torch.Tensor): A tensor of shape [batch_size, 2] representing the (x, y) coordinates of points in the batch.
    - batch_features (torch.Tensor): A tensor of shape [batch_size, num_features] representing the features.
    - grids (torch.Tensor): A tensor of shape [batch_size, channels, grid_resolution, grid_resolution].
    - x_coords (torch.Tensor): A tensor of shape [batch_size, grid_resolution] containing x coordinates for each grid cell.
    - y_coords (torch.Tensor): A tensor of shape [batch_size, grid_resolution] containing y coordinates for each grid cell.
    - full_data (np.ndarray): A numpy array representing the entire point cloud's (x, y) coordinates AND features for the KDTree.
    - channels (int): Number of feature channels in the grid (default is 3 for RGB).
    - device (torch.device): The device (CPU or GPU) to run this on.

    Returns:
    - grids (torch.Tensor): The updated grids with features assigned based on the nearest points from the full point cloud.
    """
    batch_size = batch_data.shape[0]  # Number of points in the batch
    num_available_features = batch_features.shape[1]  # How many features are available

    # Ensure we only extract up to 'channels' features
    # batch_features = batch_features[:, :min(channels, num_available_features)].to(device)

    # Build a KDTree for the full point cloud using the full_data (assumes full_data is numpy array with shape [N, 2])
    tree = KDTree(full_data[:, :2])  # We use only x, y coordinates

    # Iterate through each batch point and its grid
    for i in range(batch_size):
        # Flatten grid coordinates for the i-th batch (grid_resolution cells)
        grid_coords = torch.stack([x_coords[i], y_coords[i]], dim=1).cpu().numpy()  # Convert grid coords to numpy for KDTree search

        # Query the KDTree to find the nearest point in the full point cloud for each grid cell
        _, closest_points_idxs = tree.query(grid_coords)  # closest_points_idxs has shape [grid_resolution]

        # Assign features to the grid for the i-th batch based on the closest points from the full point cloud
        for channel in range(channels):
            for cell_idx in range(grids.shape[2]):  # Loop over grid_resolution (cells)
                # Assign the features of the closest point to the grid cell
                closest_point_idx = closest_points_idxs[cell_idx]
                # Get the features from the full point cloud, not the batch
                # Assuming full_data has shape [N, 3+num_features] where first 3 columns are (x, y, z)
                grids[i, channel, cell_idx] = full_data[
                    closest_point_idx, 3 + channel]  # Fetch the correct feature channel from full_data

    return grids


def prepare_grids_dataloader(data_array, channels, batch_size, num_workers):
    """
    Prepares the DataLoader for batching the point cloud data.

    Args:
    - data_array (np.ndarray): The dataset containing (x, y, z) coordinates, features, and class labels.
    - channels (int): Number of feature channels in the data.
    - batch_size (int): The number of points to process in each batch.
    - num_workers (int): Number of workers for data loading.

    Returns:
    - data_loader (DataLoader): A DataLoader that batches the dataset.
    """
    dataset = TensorDataset(
        torch.tensor(data_array[:, :2]),  # Center points (x, y)
        torch.tensor(data_array[:, 3:3 + channels]),  # Features
        torch.tensor(data_array[:, -1])  # Class labels
    )

    # Set num_workers > 0 to enable multiprocessing
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # careful: we cannot pass tensors to device already to avoid problems with num_workers != 0
    # so we'll need to pass the data to device after calling the dataloader!

    return data_loader


def gpu_generate_multiscale_grids(data_loader, window_sizes, grid_resolution, channels, device, full_data, save_dir=None, save=False):
    """
    Generates grids for multiple scales (small, medium, large) for the entire dataset in batches.

    Args:
    - data_loader (DataLoader): A DataLoader that batches the dataset.
    - window_sizes (list of tuples): List of window sizes for each scale (e.g., [('small', 2.5), ('medium', 5.0)]).
    - grid_resolution (int): The grid resolution (e.g., 128x128).
    - channels (int): Number of feature channels in the grid (e.g., 3 for RGB).
    - device (torch.device): The device to run on (CPU or GPU).
    - full_data (np.ndarray): The entire point cloud's data to build the KDTree for nearest neighbor search.
    - save_dir (str): Directory to save the generated grids (optional).
    - save (bool): Whether to save the grids to disk (default is False).

    Returns:
    - labeled_grids_dict (dict): Dictionary containing the generated grids and corresponding labels for each scale.
    """

    # Create a dictionary to hold grids and class labels for each scale
    labeled_grids_dict = {scale_label: {'grids': [], 'class_labels': []} for scale_label, _ in window_sizes}

    # Iterate over the DataLoader batches
    for batch_idx, (batch_data, batch_features, batch_labels) in enumerate(data_loader):
        print(f"Processing batch {batch_idx + 1}/{len(data_loader)}...")

        # Move data to the correct device (GPU or CPU)
        batch_data = batch_data.to(device)
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        # For each scale, generate the grids and assign features
        for size_label, window_size in window_sizes:
            print(f"Generating {size_label} grid for batch {batch_idx} with window size {window_size}...")

            # Create a batch of grids
            grids, _, x_coords, y_coords = gpu_create_feature_grid(batch_data, window_size, grid_resolution, channels, device)

            # Assign features to the grids
            grids = gpu_assign_features_to_grid(batch_data, batch_features, grids, x_coords, y_coords, full_data, channels, device)

            # Append the grids and labels to the dictionary
            labeled_grids_dict[size_label]['grids'].append(grids.cpu().numpy())  # Store as numpy arrays
            labeled_grids_dict[size_label]['class_labels'].append(batch_labels.cpu().numpy())

            # Save the grid if save_dir is provided
            if save and save_dir is not None:
                for i, (grid, label) in enumerate(zip(grids, batch_labels)):
                    try:
                        # Convert grid to numpy and save
                        grid_with_features = grid.cpu().numpy()

                        # Ensure the save directory exists
                        scale_dir = os.path.join(save_dir, size_label)
                        os.makedirs(scale_dir, exist_ok=True)

                        # Construct the filename and save the file
                        grid_filename = os.path.join(scale_dir,
                                                     f"grid_{batch_idx}_{i}_{size_label}_class_{int(label)}.npy")
                        np.save(grid_filename, grid_with_features)
                        print(f"Saved {size_label} grid for batch {batch_idx}, point {i} to {grid_filename}")
                    except Exception as e:
                        print(f"Error saving grid {i} in batch {batch_idx}: {str(e)}")
            elif save and save_dir is None:
                print('Warning: unspecified save directory for generated grids. Grids cannot be saved.')

    return labeled_grids_dict





