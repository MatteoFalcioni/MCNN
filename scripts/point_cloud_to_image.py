from scipy.spatial import cKDTree
from utils.point_cloud_data_utils import remap_labels
import numpy as np
import os
from tqdm import tqdm


def compute_point_cloud_bounds(data_array, padding=0.0):
    """
    Computes the spatial boundaries (min and max) of the point cloud data.
    
    Args:
    - data_array (numpy.ndarray): Array containing point cloud data where the first three columns are (x, y, z) coordinates.
    - padding (float): Optional padding to extend the boundaries by a fixed amount in all directions.
    
    Returns:
    - bounds_dict (dict): Dictionary with 'x_min', 'x_max', 'y_min', 'y_max' defining the spatial limits of the point cloud.
    """
    # Calculate the min and max values for x and y coordinates
    x_min = data_array[:, 0].min() - padding
    x_max = data_array[:, 0].max() + padding
    y_min = data_array[:, 1].min() - padding
    y_max = data_array[:, 1].max() + padding

    # Construct the boundaries dictionary
    bounds_dict = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max
    }

    return bounds_dict


def create_feature_grid(center_point, window_size, grid_resolution=128, channels=3):
    """
    Creates a grid around the center point and initializes cells to store feature values.
    Args:
    - center_point (tuple): The (x, y, z) coordinates of the center point of the grid.
    - window_size (float): The size of the square window around the center point (in meters).
    - grid_resolution (int): The number of cells in one dimension of the grid (e.g., 128 for a 128x128 grid).
    - channels (int): The number of channels in the resulting image.

    Returns:
    - grid (numpy.ndarray): A 2D grid initialized to zeros, which will store feature values.
    - cell_size (float): The size of each cell in meters.
    - x_coords (numpy.ndarray): Array of x coordinates for the centers of the grid cells.
    - y_coords (numpy.ndarray): Array of y coordinates for the centers of the grid cells.
    - constant_z (float): The fixed z coordinate for the grid cells.
    """
    # Calculate the size of each cell in meters
    cell_size = window_size / grid_resolution

    # Initialize the grid to zeros; each cell will eventually hold feature values
    grid = np.zeros((grid_resolution, grid_resolution, channels))

    # Generate cell coordinates for the grid based on the center point
    i_indices = np.arange(grid_resolution)
    j_indices = np.arange(grid_resolution)

    half_resolution_minus_half = (grid_resolution / 2) - 0.5

    # following x_k = x_pk - (64.5 - j) * w
    x_coords = center_point[0] - (half_resolution_minus_half - j_indices) * cell_size
    y_coords = center_point[1] - (half_resolution_minus_half - i_indices) * cell_size

    constant_z = center_point[2]  # Z coordinate is constant for all cells

    return grid, cell_size, x_coords, y_coords, constant_z


def assign_features_to_grid(tree, data_array, grid, x_coords, y_coords, constant_z, feature_indices):
    """
    Assigns features from the nearest point in the dataset to each cell in the grid using a pre-built KDTree.

    Args:
    - tree (KDTree): Pre-built KDTree for efficient nearest-neighbor search.
    - data_array (numpy.ndarray): Array where each row represents a point with its x, y, z coordinates and features.
    - grid (numpy.ndarray): A 2D grid initialized to zeros, which will store feature values.
    - x_coords (numpy.ndarray): Array of x coordinates for the centers of the grid cells.
    - y_coords (numpy.ndarray): Array of y coordinates for the centers of the grid cells.
    - constant_z (float): The fixed z coordinate for the grid cells.
    - feature_indices (list): List of indices for the selected features.

    Returns:
    - grid (numpy.ndarray): The grid populated with the nearest point's feature values.
    """
    
    # Generate grid coordinates for all cells in bulk
    grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')
    grid_coords = np.stack((grid_x.flatten(), grid_y.flatten(), np.full(grid_x.size, constant_z)), axis=-1)

    # Query the KDTree in bulk using all grid coordinates
    distances, indices = tree.query(grid_coords)

    # Check for valid indices or distances
    valid_mask = ~np.isinf(distances) & (indices >= 0) & (indices < len(data_array))

    # Assign features to valid cells
    valid_indices = np.where(valid_mask)[0]
    grid.flat[valid_indices] = data_array[indices[valid_indices], :][:, feature_indices].flatten()

    return grid


def generate_multiscale_grids(data_array, window_sizes, grid_resolution, features_to_use, known_features, save_dir=None, save=False):
    """
    Generates grids for each point in the data array with different window sizes, saves them to disk if required, and returns the grids.

    Args:
    - data_array (numpy.ndarray): Array where each row represents a point with its x, y, z coordinates and features.
    - window_sizes (list): List of tuples where each tuple contains (scale_label, window_size).
    - grid_resolution (int): Resolution of the grid (e.g., 128x128).
    - features_to_use (list): List of feature names to use for each grid.
    - known_features (list): List of all possible feature names in the order they appear in `data_array`.
    - save_dir (str): Directory to save the generated grids. Default is None (do not save).
    - save (bool): Whether to save the generated grids to disk. Default is False.

    Returns:
    - labeled_grids_dict (dict): A dictionary with scale labels as keys, where each entry contains 'grids' (list of grids)
      and 'class_labels' (list of corresponding class labels).
    """

    channels = len(features_to_use)  # Calculate the number of channels based on the selected features


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

    # Compute point cloud bounds
    point_cloud_bounds = compute_point_cloud_bounds(data_array)

    # Find the indices of the requested features in the known features list
    feature_indices = [known_features.index(feature) for feature in features_to_use]

    # Create KDTree once for the entire dataset with x, y, z coordinates
    tree = cKDTree(data_array[:, :3])  # Use x, y, and z coordinates for 3D KDTree

    for i in tqdm(range(num_points), desc="Generating grids", unit="grid"):
        # Select the current point as the center point for the grid
        center_point = data_array[i, :3]
        label = data_array[i, -1]  # Assuming the class label is in the last column

        for size_label, window_size in window_sizes:

            # Check if the grid centered at center_point would fall out of point cloud bounds
            half_window = window_size / 2
            if (center_point[0] - half_window < point_cloud_bounds['x_min'] or 
                center_point[0] + half_window > point_cloud_bounds['x_max'] or 
                center_point[1] - half_window < point_cloud_bounds['y_min'] or 
                center_point[1] + half_window > point_cloud_bounds['y_max']):
                # Skip generating this grid if it falls out of bounds
                # print(f"Skipping grid at center ({center_point[0]}, {center_point[1]}) for size '{size_label}' as it falls out of bounds.")
                continue
            
            # Create a grid around the current center point
            grid, _, x_coords, y_coords, z_coord = create_feature_grid(center_point, window_size, grid_resolution, channels)

            # Assign features to the grid cells using the pre-built KDTree
            grid_with_features = assign_features_to_grid(tree, data_array, grid, x_coords, y_coords, z_coord, feature_indices)

            # Transpose the grid to match PyTorch's 'channels x height x width' format
            grid_with_features = np.transpose(grid_with_features, (2, 0, 1))

            # Store the grid and label
            labeled_grids_dict[size_label]['grids'][i] = grid_with_features
            labeled_grids_dict[size_label]['class_labels'][i] = label

            # Save the grid only if it's valid
            saved = save_grid_if_valid(grid_with_features, label, i, size_label, save_dir)
            if not saved:
                continue  # Skip to the next iteration if this grid is invalid

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


def save_grid_if_valid(grid, label, point_idx, scale_label, save_dir):
    """
    Save the grid if it contains no NaN or Inf values.

    Args:
    - grid (numpy.ndarray): The grid to be saved.
    - label (int): The class label.
    - point_idx (int): Index of the point for which the grid is generated.
    - scale_label (str): The scale label ('small', 'medium', 'large').
    - save_dir (str): Directory to save the grid.
    """
    # Check for NaN or Inf in the grid
    if np.isnan(grid).any() or np.isinf(grid).any():
        print(f"Invalid grid found for {point_idx}_{scale_label}. Skipping save.")
        return False  # Skip saving if invalid

    # Convert label to an integer to avoid issues with floating point labels
    label = int(label)

    # Construct the file name and save the grid
    file_name = f"{point_idx}_{scale_label}_class_{label}.npy"
    file_path = os.path.join(save_dir, scale_label, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the grid
    np.save(file_path, grid)
    # print(f"Saved grid: {file_path}")  # Add a log statement to confirm the save

    return True


