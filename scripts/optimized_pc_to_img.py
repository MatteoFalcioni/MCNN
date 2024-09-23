import torch
from scipy.spatial import KDTree
import numpy as np
import os


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


def gpu_assign_features_to_grid(data_array, grid, x_coords, y_coords, channels=3, device=None):
    """
    Optimized feature assignment using Torch tensors and KDTree batching.
    """
    points = torch.tensor(data_array[:, :2], device=device)  # x, y coordinates in Torch
    features = torch.tensor(data_array[:, 3:3+channels], device=device)  # Features in Torch
    tree = KDTree(points.cpu().numpy())  # Build the KDTree on CPU for now

    # Iterate over each grid cell in batches
    grid_shape = grid.shape
    batch_indices = np.indices((grid_shape[1], grid_shape[2]))

    # Flatten indices to process them in a batch
    flat_indices = batch_indices.reshape(2, -1)
    flat_coords = torch.stack([x_coords[flat_indices[0]], y_coords[flat_indices[1]]], dim=1).cpu().numpy()

    # Query KDTree in a batch
    _, idxs = tree.query(flat_coords)

    # Assign features to the grid
    grid[:, flat_indices[0], flat_indices[1]] = features[idxs].T

    return grid


def gpu_generate_multiscale_grids(data_array, window_sizes, grid_resolution, channels, device, save_dir=None, save=False):
    """
    Optimized multiscale grid generation using Torch tensors.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    num_points = len(data_array)  # Number of points in the dataset

    labeled_grids_dict = {
        scale_label: {
            'grids': torch.zeros((num_points, channels, grid_resolution, grid_resolution), device=device),
            'class_labels': torch.zeros((num_points,), device=device)
        }
        for scale_label, _ in window_sizes
    }

    for i in range(num_points):
        center_point = torch.tensor(data_array[i, :3], device=device)  # (x, y, z)
        label = data_array[i, -1]  # Class label

        for size_label, window_size in window_sizes:
            print(f"Generating {size_label} grid for point {i} with window size {window_size}...")

            grid, _, x_coords, y_coords = gpu_create_feature_grid(center_point, window_size, grid_resolution, channels, device)

            grid_with_features = gpu_assign_features_to_grid(data_array, grid, x_coords, y_coords, channels, device)

            # Store the grid and label
            labeled_grids_dict[size_label]['grids'][i] = grid_with_features
            labeled_grids_dict[size_label]['class_labels'][i] = label

            if save and save_dir is not None:
                grid_with_features_np = grid_with_features.cpu().numpy()  # Move back to CPU for saving
                scale_dir = os.path.join(save_dir, size_label)
                os.makedirs(scale_dir, exist_ok=True)
                grid_filename = os.path.join(scale_dir, f"grid_{i}_{size_label}_class_{int(label)}.npy")
                np.save(grid_filename, grid_with_features_np)
                print(f"Saved {size_label} grid for point {i} to {grid_filename}")

    return labeled_grids_dict
