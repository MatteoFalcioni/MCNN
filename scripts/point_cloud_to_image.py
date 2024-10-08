from scipy.spatial import cKDTree
from utils.point_cloud_data_utils import remap_labels, save_features_used
import numpy as np
import os
from tqdm import tqdm
import csv


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
    _, indices = tree.query(grid_coords)

    # Assign features using the bulk indices
    grid[:, :, :] = data_array[indices, :][:, feature_indices].reshape(grid.shape)

    return grid


def generate_multiscale_grids(data_array, window_sizes, grid_resolution, features_to_use, known_features, save_dir=None):
    """
    Generates multiscale grids for each point in the data array, saves the grids to disk in .npy format, and saves the class labels separately in a CSV file.

    Args:
    - data_array (numpy.ndarray): A 2D array where each row represents a point in the point cloud, with its x, y, z coordinates and features.
    - window_sizes (list): A list of tuples where each tuple contains (scale_label, window_size). Example: [('small', 2.5), ('medium', 5.0), ('large', 10.0)].
    - grid_resolution (int): The resolution of the grid (e.g., 128 for 128x128 grids).
    - features_to_use (list): A list of feature names (strings) to use for each grid (e.g., ['intensity', 'R', 'G', 'B']).
    - known_features (list): A list of all possible feature names in the order they appear in `data_array`. Used to extract the indices of the features to use.
    - save_dir (str): The directory where the generated grids, labels and used features will be saved. Default is None. 

    Returns:
    - None, just saves the grids and labels to files.

    Behavior:
    - For each point in the data array, multiscale grids are generated using the provided window sizes and features.
    - The grids are saved to disk in .npy format, without class labels embedded in the filenames.
    - Class labels are saved separately in a CSV file for each scale (e.g., small_labels.csv, medium_labels.csv, large_labels.csv).

    Example:
    If the function is called with window sizes [('small', 2.5), ('medium', 5.0), ('large', 10.0)], and save_dir is set to 'data/preprocessed/', the output directory will look like this:
    
    - data/preprocessed/small/
        - 0_small.npy
        - 1_small.npy
        - ...
        - small_labels.csv (contains the point indices and class labels)
    
    - data/preprocessed/medium/
        - 0_medium.npy
        - 1_medium.npy
        - ...
        - medium_labels.csv

    - data/preprocessed/large/
        - 0_large.npy
        - 1_large.npy
        - ...
        - large_labels.csv
    """

    channels = len(features_to_use)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    else:
        raise ValueError('Error: no specified directory to save grids.')

    # Remap labels to continuous integers (needed for cross-entropy loss)
    data_array, _ = remap_labels(data_array)

    num_points = len(data_array)

    point_cloud_bounds = compute_point_cloud_bounds(data_array)
    feature_indices = [known_features.index(feature) for feature in features_to_use]    # get feature indices (not order-dependant)
    save_features_used(features_to_use, save_dir)   # save used feature for inference or successive loading of files

    tree = cKDTree(data_array[:, :3])

    for size_label, window_size in window_sizes:

        # Open the CSV file once per scale to save labels
        label_file_path = os.path.join(save_dir, f"{size_label}_labels.csv")

        with open(label_file_path, mode='w', newline='') as label_file:
            label_writer = csv.writer(label_file)
            label_writer.writerow(['point_idx', 'label'])  # Write header

            for i in tqdm(range(num_points), desc=f"Generating {size_label} grids", unit="grid"):
                center_point = data_array[i, :3]
                label = data_array[i, -1]

                half_window = window_size / 2
                if (center_point[0] - half_window < point_cloud_bounds['x_min'] or 
                    center_point[0] + half_window > point_cloud_bounds['x_max'] or 
                    center_point[1] - half_window < point_cloud_bounds['y_min'] or 
                    center_point[1] + half_window > point_cloud_bounds['y_max']):
                    continue  # Skip grids that fall out of bounds

                grid, _, x_coords, y_coords, z_coord = create_feature_grid(center_point, window_size, grid_resolution, channels)
                grid_with_features = assign_features_to_grid(tree, data_array, grid, x_coords, y_coords, z_coord, feature_indices)
                grid_with_features = np.transpose(grid_with_features, (2, 0, 1))    # save grid in Torch format (channels first)

                # Save the grid and write the label directly to CSV
                save_grid_if_valid(grid_with_features, label, i, size_label, save_dir, label_writer)

    print('Multiscale grid generation and label saving completed successfully.')

    return 


def save_grid_if_valid(grid, label, point_idx, scale_label, save_dir, label_writer=None):
    """
    Save the grid as .npy if it contains no NaN or Inf values, and write the corresponding label to a CSV.

    Args:
    - grid (numpy.ndarray): The grid to be saved.
    - label (int): The class label.
    - point_idx (int): Index of the point for which the grid is generated.
    - scale_label (str): The scale label ('small', 'medium', 'large').
    - save_dir (str): Directory to save the grid.
    - label_writer (csv.writer): CSV writer to directly write labels.
    """
    # Check for NaN or Inf in the grid
    if np.isnan(grid).any() or np.isinf(grid).any():
        print(f"Invalid grid found for {point_idx}_{scale_label}. Skipping save.")
        return False  # Skip saving if invalid

    # Construct the file name (now without embedding the class label)
    file_name = f"{point_idx}_{scale_label}.npy"
    file_path = os.path.join(save_dir, scale_label, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the grid
    np.save(file_path, grid)

    # Write the label directly to the CSV
    if label_writer:
        label_writer.writerow([point_idx, label])

    return True


