from scipy.spatial import cKDTree
from utils.point_cloud_data_utils import remap_labels, save_features_used
import numpy as np
import os
from tqdm import tqdm
import csv
import pandas as pd
import zipfile
import io
import subprocess
import tempfile


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


def generate_multiscale_grids(center_point, data_array, window_sizes, grid_resolution, feature_indices, kdtree, point_cloud_bounds):
    """
    Generates multiscale grids for a single point in the data array. 
    
    Args:
    - center_point (numpy.ndarray): (x, y, z) coordinates of the point in the point cloud.
    - data_array (numpy.ndarray): 2D array containing the point cloud data.
    - window_sizes (list): List of tuples where each tuple contains (scale_label, window_size). 
                           Example: [('small', 2.5), ('medium', 5.0), ('large', 10.0)].
    - grid_resolution (int): Resolution of the grid (e.g., 128 for 128x128 grids).
    - feature_indices (list): List of feature indices to be selected from the full list of features. 
    - kdtree (KDTree): Prebuilt KDTree for nearest neighbor search.
    - point_cloud_bounds (dict): Dictionary containing point cloud boundaries in every dimension (x, y, z).
    
    Returns:
    - grids_dict (dict): Dictionary of generated grids for each scale.
    """
    
    grids_dict = {}  # To store grids for each scale
    channels = len(feature_indices)
    

    for size_label, window_size in window_sizes:
        half_window = window_size / 2

        # Check if the point is within bounds
        if (center_point[0] - half_window < point_cloud_bounds['x_min'] or
            center_point[0] + half_window > point_cloud_bounds['x_max'] or
            center_point[1] - half_window < point_cloud_bounds['y_min'] or
            center_point[1] + half_window > point_cloud_bounds['y_max']):
            # skip grid generation for all scales if one scale is invalid
            break

        # Generate the grid for the current scale
        grid, _, x_coords, y_coords, z_coord = create_feature_grid(
            center_point, window_size, grid_resolution, channels
        )

        # Assign features from the nearest point in the data array using the KDTree
        grid_with_features = assign_features_to_grid(kdtree, data_array, grid, x_coords, y_coords, z_coord, feature_indices)
        grid_with_features = np.transpose(grid_with_features, (2, 0, 1))  # Convert to channels first (Torch format)

        # Check for NaN or Inf in the grid
        if np.isnan(grid_with_features).any() or np.isinf(grid_with_features).any():
            # skip grid generation for all scales if one scale is invalid
            break

        grids_dict[size_label] = grid_with_features

    return grids_dict












'''def generate_multiscale_grids(data_array, window_sizes, grid_resolution, features_to_use, known_features, save_dir=None):
    """
    Generates multiscale grids for each point in the data array, saves the grids to disk in a zipped .npy format, 
    and saves the class labels in a single CSV file.

    Args:
    - data_array (numpy.ndarray): 2D array where each row represents a point in the point cloud, with its x, y, z coordinates and features.
    - window_sizes (list): List of tuples where each tuple contains (scale_label, window_size). 
                           Example: [('small', 2.5), ('medium', 5.0), ('large', 10.0)].
    - grid_resolution (int): Resolution of the grid (e.g., 128 for 128x128 grids).
    - features_to_use (list): List of feature names (strings) to use for each grid (e.g., ['intensity', 'red', 'green', 'blue']).
    - known_features (list): List of all possible feature names in the order they appear in `data_array`.
    - save_dir (str): Directory where the generated grids and labels will be saved.

    Returns:
    - None. Grids and labels are saved to files.

    Behavior:
    - For each point in the data array, multiscale grids are generated using the provided window sizes and features.
    - The grids are saved to disk in .npy format, without class labels embedded in the filenames.
    - A single class labels file (`labels.csv`) is saved for all scales.
    """

    channels = len(features_to_use)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    else:
        raise ValueError('Error: no specified directory to save grids.')

    # Remap labels to continuous integers (needed for cross-entropy loss)
    data_array, _ = remap_labels(data_array)

    num_points = len(data_array)

    point_cloud_bounds = compute_point_cloud_bounds(data_array) # get point cloud bounds
    feature_indices = [known_features.index(feature) for feature in features_to_use]  # Get feature indices
    save_features_used(features_to_use, save_dir)  # Save used features for future inference or loading

    tree = cKDTree(data_array[:, :3])  # Prebuild the KDTree

    # Open a single ZIP file for grids and CSV for labels
    zip_file_path = os.path.join(save_dir, "grids.zip")
    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        label_file_path = os.path.join(save_dir, "labels.csv")
        
        with open(label_file_path, mode='w', newline='') as label_file:
            label_writer = csv.writer(label_file)
            label_writer.writerow(['point_idx', 'label'])  # Write header
            
            skipped_nan_grids = 0
            skipped_out_of_bounds = 0

            for i in tqdm(range(num_points), desc="Generating multiscale grids", unit="grid"):

                # Track whether all grids for this point are valid
                all_grids_valid = True
                grids_to_save = {}

                center_point = data_array[i, :3]
                label = data_array[i, -1]

                # Loop through all scales and generate grids
                for size_label, window_size in window_sizes:

                    half_window = window_size / 2
                    if (center_point[0] - half_window < point_cloud_bounds['x_min'] or
                        center_point[0] + half_window > point_cloud_bounds['x_max'] or
                        center_point[1] - half_window < point_cloud_bounds['y_min'] or
                        center_point[1] + half_window > point_cloud_bounds['y_max']):
                        all_grids_valid = False
                        skipped_out_of_bounds += 1
                        break  # If the grid is out-of-bounds, skip this point for all scales

                    # Generate the grid for the current scale
                    grid, _, x_coords, y_coords, z_coord = create_feature_grid(
                        center_point, window_size, grid_resolution, channels
                    )
                    grid_with_features = assign_features_to_grid(tree, data_array, grid, x_coords, y_coords, z_coord, feature_indices)
                    grid_with_features = np.transpose(grid_with_features, (2, 0, 1))  # Convert grid to Torch format (channels first)

                    # Check if the grid is valid (no NaN or Inf)
                    if np.isnan(grid_with_features).any() or np.isinf(grid_with_features).any():
                        skipped_nan_grids += 1
                        all_grids_valid = False
                        break  # If any grid is invalid, skip this point for all scales

                    # Temporarily store the valid grid for this scale (for later saving if all are valid)
                    grids_to_save[size_label] = grid_with_features

                # If all grids are valid, save them and write the label
                if all_grids_valid:
                    for size_label, grid in grids_to_save.items():
                        # Save grids as bytes in a ZIP file
                        grid_name = f"{i}_{size_label}.npy"
                        np_bytes = numpy_to_bytes(grid)  # Convert NumPy array to bytes
                        zip_file.writestr(grid_name, np_bytes)

                    # Write label to CSV
                    label_writer.writerow([i, label])
                
                

        print('Multiscale grid generation and single label file saving completed successfully.')
        print(f'{skipped_nan_grids} were skipped because of nan or inf values, {skipped_out_of_bounds} were skipped because out of bounds')'''


'''def load_saved_grids(grid_save_dir):
    """
    Loads saved grid file paths (filenames) and corresponding labels from a ZIP file and a single CSV file.

    Args:
    - grid_save_dir (str): Directory where the ZIP file and CSV are saved.

    Returns:
    - grids_dict (dict): Dictionary containing the filenames for each grid (by scale).
    - labels (list): List of labels corresponding to the grids (loaded from CSV).
    """
    grids_dict = {'small': [], 'medium': [], 'large': []}
    labels = []

    # Open the ZIP file and read the list of filenames
    zip_file_path = os.path.join(grid_save_dir, "grids.zip")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        # Collect the filenames for each scale
        grids_dict['small'] = [f for f in zip_file.namelist() if '_small.npy' in f]
        grids_dict['medium'] = [f for f in zip_file.namelist() if '_medium.npy' in f]
        grids_dict['large'] = [f for f in zip_file.namelist() if '_large.npy' in f]

    # Load the labels from the unified 'labels.csv' file
    labels_file = os.path.join(grid_save_dir, 'labels.csv')
    labels_df = pd.read_csv(labels_file)
    
    # Extract the labels
    labels = labels_df['label'].tolist()

    # Ensure we have the same number of grids and labels
    num_small = len(grids_dict['small'])
    num_medium = len(grids_dict['medium'])
    num_large = len(grids_dict['large'])
    num_labels = len(labels)

    if not (num_small == num_medium == num_large):
        raise ValueError(f"Inconsistent number of grids across scales: small={num_small}, medium={num_medium}, large={num_large}")

    if not (num_small == num_labels):
        raise ValueError(f"Inconsistent number of grids and labels: small={num_small}, labels={num_labels}")

    print(f"Loaded {num_small} filenames for each scale with {num_labels} labels.")

    return grids_dict, labels'''
