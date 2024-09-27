from scipy.spatial import KDTree
from utils.point_cloud_data_utils import remap_labels
import numpy as np
import os
from tqdm import tqdm


def create_feature_grid(center_point, window_size, grid_resolution=128, channels=3):
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
    - z_coords (numpy.ndarray): Array of z coordinates for the centers of the grid cells.
    """
    # Calculate the size of each cell in meters
    cell_size = window_size / grid_resolution

    # Initialize the grid to zeros; each cell will eventually hold feature values
    grid = np.zeros((grid_resolution, grid_resolution, channels))

    # Generate cell coordinates for the grid based on the center point
    i_indices = np.arange(grid_resolution)
    j_indices = np.arange(grid_resolution)

    half_resolution_plus_half = (grid_resolution / 2) + 0.5

    # following x_k = x_pk - (64.5 - j) * w
    x_coords = center_point[0] - (half_resolution_plus_half - j_indices) * cell_size
    y_coords = center_point[1] - (half_resolution_plus_half - i_indices) * cell_size
    z_coords = np.full((grid_resolution, grid_resolution), center_point[2])  # Z coordinate is constant for all cells

    return grid, cell_size, x_coords, y_coords, z_coords


def assign_features_to_grid(tree, data_array, grid, x_coords, y_coords, channels=3):
    """
    Assigns features from the nearest point in the dataset to each cell in the grid using a pre-built KDTree.

    Args:
    - tree (KDTree): Pre-built KDTree for efficient nearest-neighbor search.
    - data_array (numpy.ndarray): Array where each row represents a point with its x, y, z coordinates and features.
    - grid (numpy.ndarray): A 2D grid initialized to zeros, which will store feature values.
    - x_coords (numpy.ndarray): Array of x coordinates for the centers of the grid cells.
    - y_coords (numpy.ndarray): Array of y coordinates for the centers of the grid cells.
    - channels (int): Number of feature channels to assign to each grid cell (default is 3 for RGB).

    Returns:
    - grid (numpy.ndarray): The grid populated with the nearest point's feature values.
    """
    # Iterate over each cell in the grid
    for i in range(len(x_coords)):
        for j in range(len(y_coords)):
            # Find the nearest point to the cell center (x_coords[i], y_coords[j])
            dist, idx = tree.query([x_coords[i], y_coords[j]])

            # Assign the features of the nearest point to the grid cell
            grid[i, j, :channels] = data_array[idx, 3:3 + channels]  # Assuming features start from the 4th column (index 3)

    return grid


def generate_multiscale_grids(data_array, window_sizes, grid_resolution, channels, save_dir=None, save=False):
    """
    Generates grids for each point in the data array with different window sizes, saves them to disk if required, and returns the grids.

    Args:
    - data_array (numpy.ndarray): Array where each row represents a point with its x, y, z coordinates and features.
    - window_sizes (list): List of tuples where each tuple contains (scale_label, window_size).
    - grid_resolution (int): Resolution of the grid (e.g., 128x128).
    - channels (int): Number of channels to store in each grid.
    - save_dir (str): Directory to save the generated grids. Default is None (do not save).
    - save (bool): Whether to save the generated grids to disk. Default is False.

    Returns:
    - labeled_grids_dict (dict): A dictionary with scale labels as keys, where each entry contains 'grids' (list of grids)
      and 'class_labels' (list of corresponding class labels).
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Remap labels to continuous integers (needed for cross-entropy loss)
    data_array, _ = remap_labels(data_array)

    # Initialize a dictionary to store the generated grids and labels by window size (grids in Torch standard formats, channel first)
    num_points = len(data_array)
    labeled_grids_dict = {
        scale_label: {
            'grids': np.zeros((num_points, channels, grid_resolution, grid_resolution)),  # Channel-first
            'class_labels': np.zeros((num_points,))
        }
        for scale_label, _ in window_sizes
    }

    # Create KDTree once for the entire dataset
    tree = KDTree(data_array[:, :2])  # Use x, y coordinates for 2D grid

    for i in tqdm(range(num_points), desc="Generating grids", unit="grid"):
        # Select the current point as the center point for the grid
        center_point = data_array[i, :3]
        label = data_array[i, -1]  # Assuming the class label is in the last column

        for size_label, window_size in window_sizes:
            # print(f"Generating {size_label} grid for point {i} with window size {window_size}...")

            # Create a grid around the current center point
            grid, _, x_coords, y_coords, _ = create_feature_grid(center_point, window_size, grid_resolution, channels)

            # Assign features to the grid cells using the pre-built KDTree
            grid_with_features = assign_features_to_grid(tree, data_array, grid, x_coords, y_coords, channels)

            # Transpose the grid to match PyTorch's 'channels x height x width' format
            grid_with_features = np.transpose(grid_with_features, (2, 0, 1))

            # Store the grid and label
            labeled_grids_dict[size_label]['grids'][i] = grid_with_features
            labeled_grids_dict[size_label]['class_labels'][i] = label

            # Save the grid if required
            if save and save_dir is not None:
               save_grid(grid_with_features, label, i, size_label, save_dir)

    print('Multiscale grid generation completed successfully.')

    return labeled_grids_dict


def save_grid(grid, label, point_idx, size_label, save_dir):
    """
    Helper function to save a grid to disk.

    Args:
    - grid (numpy.ndarray): The grid to be saved.
    - label (int): The class label corresponding to the grid.
    - point_idx (int): Index of the point for which the grid is generated.
    - size_label (str): Label for the grid scale ('small', 'medium', 'large').
    - save_dir (str): Directory to save the generated grids.
    """
    try:
        scale_dir = os.path.join(save_dir, size_label)
        os.makedirs(scale_dir, exist_ok=True)
        grid_filename = os.path.join(scale_dir, f"grid_{point_idx}_{size_label}_class_{int(label)}.npy")
        np.save(grid_filename, grid)
    except Exception as e:
        print(f"Error saving grid for point {point_idx} in {size_label} scale: {str(e)}")





