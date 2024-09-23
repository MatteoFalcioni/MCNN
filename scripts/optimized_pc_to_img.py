import torch
from scipy.spatial import KDTree
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset


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

    print("Shape of features:", features.shape)
    print("Shape of idxs:", idxs.shape)
    print("Shape of grid before assignment:", grid.shape)

    # Assign features to the grid
    if len(idxs) > 0:
        grid[:, flat_indices[0], flat_indices[1]] = features[idxs].T
    else:
        print("Warning: features[idxs] empty.")


    return grid


def batch_process(data_loader, window_sizes, grid_resolution, channels, device, save_dir=None, save=False):
    """
    Batch processing for GPU-accelerated grid generation.
    """
    labeled_grids_dict = {scale_label: {'grids': [], 'class_labels': []} for scale_label, _ in window_sizes}

    for batch_data, batch_labels in data_loader:
        for i, data_point in enumerate(batch_data):
            center_point = data_point[:3].to(device)
            label = batch_labels[i].to(device)

            for size_label, window_size in window_sizes:
                grid, _, x_coords, y_coords = gpu_create_feature_grid(center_point, window_size, grid_resolution, channels, device)
                grid_with_features = gpu_assign_features_to_grid(batch_data.cpu().numpy(), grid, x_coords, y_coords, channels, device)

                labeled_grids_dict[size_label]['grids'].append(grid_with_features)
                labeled_grids_dict[size_label]['class_labels'].append(label)

                if save and save_dir is not None:
                    grid_with_features_np = grid_with_features.cpu().numpy()
                    scale_dir = os.path.join(save_dir, size_label)
                    os.makedirs(scale_dir, exist_ok=True)
                    grid_filename = os.path.join(scale_dir, f"grid_{i}_{size_label}_class_{int(label)}.npy")
                    np.save(grid_filename, grid_with_features_np)

    return labeled_grids_dict


def gpu_generate_multiscale_grids(data_array, window_sizes, grid_resolution, channels, device, save_dir=None, save=False, batch_size=50, num_workers=4):
    """
    Optimized multiscale grid generation using Torch tensors with parallel batching.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Convert data_array into a TensorDataset for batching
    dataset = TensorDataset(torch.tensor(data_array[:, :3]), torch.tensor(data_array[:, -1]))  # Assuming last column is class labels
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return batch_process(data_loader, window_sizes, grid_resolution, channels, device, save_dir, save)
