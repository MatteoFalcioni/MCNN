import torch
from scipy.spatial import KDTree
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from utils.point_cloud_data_utils import remap_labels


def gpu_create_feature_grid(center_point, window_size, grid_resolution=128, channels=3, device=None):
    """
    Optimized to work with Torch tensors for GPU acceleration.
    """
    # Calculate the size of each cell in meters
    cell_size = window_size / grid_resolution

    # Initialize the grid with zeros using Torch tensors
    grid = torch.zeros((channels, grid_resolution, grid_resolution), device=device)

    # Calculate coordinates using Torch tensors
    half_resolution_plus_half = (grid_resolution / 2) + 0.5

    x_coords = center_point[0] - (half_resolution_plus_half - torch.arange(grid_resolution, device=device)) * cell_size
    y_coords = center_point[1] - (half_resolution_plus_half - torch.arange(grid_resolution, device=device)) * cell_size

    return grid, cell_size, x_coords, y_coords


def gpu_assign_features_to_grid(batch_data, features, grid, x_coords, y_coords, channels=3, device=None):
    """
    Optimized feature assignment using Torch tensors and KDTree batching.
    Now accepts 'features' as a separate parameter.
    """
    points = batch_data[:, :2]  # x, y coordinates in Torch

    print(f"Features shape: {features.shape}")
    num_available_features = features.shape[1]  # Compute how many features are available
    print(f'number of available features: {num_available_features}')

    # Ensure we only extract up to 'channels' features
    features = features[:, :min(channels, num_available_features)]  # Use passed 'features'

    print(f"Data points shape: {points.shape}")
    print(f"Features shape: {features.shape}")

    # Only convert to CPU when necessary for KDTree
    points_cpu = points.cpu().numpy()  # Convert just for KDTree
    tree = KDTree(points_cpu)

    # Iterate over each grid cell in batches

    # following needed for debug, can be removed when all working...
    grid_shape = grid.shape
    batch_indices = np.indices((grid_shape[1], grid_shape[2]))
    # ...up to here

    # Flatten indices to process them in a batch
    flat_indices = batch_indices.reshape(2, -1)
    flat_coords = torch.stack([x_coords[flat_indices[0]], y_coords[flat_indices[1]]], dim=1).cpu().numpy()

    # Query KDTree in a batch
    _, idxs = tree.query(flat_coords)

    print("Shape of features:", features.shape)
    print("Shape of idxs:", idxs.shape)
    print("Shape of grid before assignment:", grid.shape)

    if len(idxs) > 0:
        # Check for valid indices
        valid_idxs = (idxs >= 0) & (idxs < len(features))  # Ensure we aren't indexing out of bounds
        if not valid_idxs.all():
            print(f"Warning: Some indices are out of bounds. Valid indices: {valid_idxs.sum()} / {len(idxs)}")

        print(f"Shape of grid before feature assignment: {grid.shape}")

        # Assign features to the grid
        try:
            grid[:, flat_indices[0], flat_indices[1]] = features[idxs].T
        except Exception as e:
            print(f"Error during feature assignment: {e}")
            print(f"Features selected: {features[idxs].T}")
            print(f"Grid shape: {grid.shape}")
            raise
    else:
        print("Warning: features[idxs] empty.")

    return grid


def batch_process(data_loader, window_sizes, grid_resolution, channels, device, save_dir=None, save=False):
    """
    Batch processing for GPU-accelerated grid generation.
    """
    labeled_grids_dict = {scale_label: {'grids': [], 'class_labels': []} for scale_label, _ in window_sizes}

    for batch_idx, (batch_data, batch_features, batch_labels) in enumerate(data_loader):
        print(f"Processing batch {batch_idx + 1}/{len(data_loader)} with {len(batch_data)} points")

        # Process each point, its features, and its label together
        for data_point, features, label in zip(batch_data, batch_features, batch_labels):
            center_point = data_point.to(device)  # data_point has (x, y, z)
            features = features.to(device)  # Corresponding features
            label = label.to(device)  # Corresponding label

            for size_label, window_size in window_sizes:
                print(f"Generating {size_label} grid for point {i} with window size {window_size}...")

                grid, _, x_coords, y_coords = gpu_create_feature_grid(center_point, window_size, grid_resolution,
                                                                      channels, device)

                # Now pass both batch_data (for coordinates) and features to the assign function
                # Pass the data as is (still on GPU)
                grid_with_features = gpu_assign_features_to_grid(batch_data, features, grid, x_coords, y_coords,
                                                                 channels, device)

                labeled_grids_dict[size_label]['grids'].append(grid_with_features)
                labeled_grids_dict[size_label]['class_labels'].append(label)

                if save and save_dir is not None:
                    grid_with_features_np = grid_with_features.cpu().numpy()
                    scale_dir = os.path.join(save_dir, size_label)
                    os.makedirs(scale_dir, exist_ok=True)
                    grid_filename = os.path.join(scale_dir, f"grid_{i}_{size_label}_class_{int(label)}.npy")
                    np.save(grid_filename, grid_with_features_np)
                    print(f"Saved {size_label} grid for point {i} to {grid_filename}")

    return labeled_grids_dict


def gpu_generate_multiscale_grids(data_array, window_sizes, grid_resolution, channels, device, save_dir=None, save=False, batch_size=50, num_workers=4):
    """
    Optimized multiscale grid generation using Torch tensors with parallel batching.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    data_array, _ = remap_labels(data_array)

    # Include the features along with the coordinates (x, y, z) in the dataset
    dataset = TensorDataset(torch.tensor(data_array[:, :3]), torch.tensor(data_array[:, 3:3 + channels]), torch.tensor(data_array[:, -1]))  # Including class labels as well
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return batch_process(data_loader, window_sizes, grid_resolution, channels, device, save_dir, save)
