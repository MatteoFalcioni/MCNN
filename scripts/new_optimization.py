from scipy.spatial import KDTree
from utils.point_cloud_data_utils import remap_labels
import numpy as np
import os
import torch


def opt_create_feature_grid(center_point, window_size, grid_resolution=128, channels=3):
    """
    Creates a grid around the center point and initializes cells to store feature values.

    Args:
    - center_point (tuple): The (x, y, z) coordinates of the center point of the grid.
    - window_size (float): The size of the square window around the center point (in meters).
    - grid_resolution (int): The number of cells in one dimension of the grid (e.g., 128 for a 128x128 grid). Default is 128 to match article to replicate.
    - channels (int): The number of channels in the resulting image. Default is 3 for RGB.

    Returns:
    - grid (numpy.ndarray): A 2D grid initialized to zeros, which will store feature values.
    - cell_size (float): The size of each cell in meters.
    - x_coords (numpy.ndarray): Array of x coordinates for the centers of the grid cells.
    - y_coords (numpy.ndarray): Array of y coordinates for the centers of the grid cells.
    """
    # Calculate the size of each cell in meters
    cell_size = window_size / grid_resolution

    # Initialize the grid to zeros; each cell will eventually hold feature values
    grid = []

    # Generate cell coordinates for the grid based on the center point
    i_indices = np.arange(grid_resolution)
    j_indices = np.arange(grid_resolution)

    half_resolution_plus_half = (grid_resolution / 2) + 0.5

    # following x_k = x_pk - (64.5 - j) * w
    x_coords = center_point[0] - (half_resolution_plus_half - j_indices) * cell_size
    y_coords = center_point[1] - (half_resolution_plus_half - i_indices) * cell_size

    return grid, cell_size, x_coords, y_coords


def opt_assign_features_to_grid(data_array, grid, x_coords, y_coords, channels=3, device='cuda'):
    """
    Assign features from the nearest point to each cell in the grid using KDTree (CPU) and assign features on GPU.

    Args:
    - data_array (numpy.ndarray): Array where each row represents a point with its x, y, z coordinates and features.
    - grid (torch.Tensor): A pre-allocated grid tensor.
    - x_coords (numpy.ndarray): Array of x coordinates for the centers of the grid cells.
    - y_coords (numpy.ndarray): Array of y coordinates for the centers of the grid cells.
    - channels (int): Number of feature channels to assign to each grid cell (default is 3 for RGB).
    - device (str): The device to perform computations on ('cuda' for GPU).

    Returns:
    - grid (torch.Tensor): Grid populated with feature values (on GPU).
    """
    # Move the grid to GPU
    grid = grid.to(device)

    # Extract point coordinates (x, y) for KDTree
    points = data_array[:, :2]  # Assuming x, y are the first two columns

    # Create a KDTree for efficient nearest-neighbor search
    tree = KDTree(points)

    # Combine x_coords and y_coords for batch querying
    grid_centers = np.array([[x, y] for x in x_coords for y in y_coords])

    # Query all grid centers at once (batch query)
    dists, indices = tree.query(grid_centers)

    # Assign the features to the grid dynamically
    for idx, (dist, nearest_idx) in enumerate(zip(dists, indices)):
        i = idx // len(y_coords)  # Calculate i and j from flat index
        j = idx % len(y_coords)

        # Get the feature vector of the nearest point
        feature_vector = torch.tensor(data_array[nearest_idx, 3:3 + channels], device=device)

        # Assign the features directly to the grid
        grid[i, j, :channels] = feature_vector

    return grid


def opt_generate_multiscale_grids(data_array, window_sizes, grid_resolution, channels, save_dir=None, save=False, device='cuda'):
    """
    Generates grids for each point in the data array with different window sizes and saves them to disk.
    Returns a dictionary with grids and their corresponding labels for each window size.
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # remap labels to continuous integers (needed for cross entropy loss)
    data_array, _ = remap_labels(data_array)

    # Initialize a dictionary to store the generated grids and labels by window size
    num_points = len(data_array)

    labeled_grids_dict = {
        scale_label: {
            'grids': torch.zeros((num_points, grid_resolution, grid_resolution, channels), device=device),  # Pre-allocate tensor
            'class_labels': np.zeros((num_points,))  # Class labels remain in CPU
        }
        for scale_label, _ in window_sizes
    }

    for i in range(num_points):
        # Select the current point as the center point for the grid
        center_point = data_array[i, :3]  # (x, y, z)
        label = data_array[i, -1]  # Assuming the class label is the last column

        for size_label, window_size in window_sizes:
            print(f"Generating {size_label} grid for point {i} with window size {window_size}...")

            # Create a grid around the current center point
            grid, _, x_coords, y_coords = create_feature_grid(center_point, window_size, grid_resolution, channels)

            # Pre-allocate grid for features
            grid_tensor = torch.zeros((grid_resolution, grid_resolution, channels), device=device)

            # Assign features to the grid cells (with GPU support)
            grid_with_features = opt_assign_features_to_grid(data_array, grid_tensor, x_coords, y_coords, channels, device=device)

            # Store the grid and label in the PyTorch tensors
            labeled_grids_dict[size_label]['grids'][i] = grid_with_features
            labeled_grids_dict[size_label]['class_labels'][i] = label

            # Save the grid if required
            if save and save_dir is not None:
                grid_with_features_cpu = grid_with_features.cpu().numpy()
                grid_with_features_cpu = np.transpose(grid_with_features_cpu, (2, 0, 1))

                scale_dir = os.path.join(save_dir, size_label)
                os.makedirs(scale_dir, exist_ok=True)
                grid_filename = os.path.join(scale_dir, f"grid_{i}_{size_label}_class_{int(label)}.npy")
                np.save(grid_filename, grid_with_features_cpu)
                print(f"Saved {size_label} grid for point {i} to {grid_filename}")

    return labeled_grids_dict



