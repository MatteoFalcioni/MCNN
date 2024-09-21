from scipy.spatial import KDTree
import numpy as np
import os


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


def assign_features_to_grid(data_array, grid, x_coords, y_coords, channels=3):
    """
    Assign features from the nearest point to each cell in the grid.

    Args:
    - data_array (numpy.ndarray): Array where each row represents a point with its x, y, z coordinates and features.
    - grid (numpy.ndarray): A 2D grid initialized to zeros, which will store feature values.
    - x_coords (numpy.ndarray): Array of x coordinates for the centers of the grid cells.
    - y_coords (numpy.ndarray): Array of y coordinates for the centers of the grid cells.
    - channels (int): Number of feature channels to assign to each grid cell (default is 3 for RGB).

    Returns:
    - grid (numpy.ndarray): Grid populated with feature values.
    """
    # Extract point coordinates (x, y) for KDTree
    points = data_array[:, :2]  # Assuming x, y are the first two columns

    # Create a KDTree for efficient nearest-neighbor search
    tree = KDTree(points)  # Only use x, y for 2D grid distance calculation

    # Iterate over each cell in the grid
    for i in range(len(x_coords)):
        for j in range(len(y_coords)):
            # Find the nearest point to the cell center (x_coords[i], y_coords[j])
            dist, idx = tree.query([x_coords[i], y_coords[j]])

            # Assign the features of the nearest point to the grid cell, considering the specified number of channels
            grid[i, j, :channels] = data_array[idx,
                                    3:3 + channels]  # Assuming features start from the 4th column (index 3)
    return grid


def generate_multiscale_grids(data_array, window_sizes, grid_resolution, channels, save_dir, save=True):
    """
    Generates grids for each point in the data array with different window sizes and saves them to disk.
    Returns a dictionary with grids and their corresponding labels for each window size.

    Args:
    - data_array (np.ndarray): Array where each row represents a point with its x, y, z coordinates and features.
    - window_sizes (list): List of window sizes to use for grid generation (e.g., [small, medium, large]).
    - grid_resolution (int): Resolution of the grid (e.g., 128x128).
    - channels (int): Number of channels to store in each grid.
    - save_dir (str): Directory to save the generated grids.
    - save (bool): Boolean value to save or discard the generated grids. Default is True.

    Returns:
    - labeled_grids_dict (dict): A dictionary with window size labels as keys. Each entry contains a dictionary
                          with 'grids' and 'labels' keys, where 'grids' is a list of generated grids and
                          'labels' is a list of corresponding class labels.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Initialize a dictionary to store the generated grids and labels by window size
    labeled_grids_dict = {scale_label: {'grids': [], 'class_labels': []} for scale_label, _ in window_sizes}

    for i in range(len(data_array)):
        # Select the current point as the center point for the grid
        center_point = data_array[i, :3]  # (x, y, z)
        label = data_array[i, -1]  # Assuming the class label is the last column

        for size_label, window_size in window_sizes:
            print(f"Generating {size_label} grid for point {i} with window size {window_size}...")

            # Create a grid around the current center point
            grid, _, x_coords, y_coords, _ = create_feature_grid(center_point, window_size, grid_resolution, channels)

            # Assign features to the grid cells
            grid_with_features = assign_features_to_grid(data_array, grid, x_coords, y_coords, channels)

            # Append the grid and the label to the respective lists in the dictionary
            labeled_grids_dict[size_label]['grids'].append(grid_with_features)
            labeled_grids_dict[size_label]['class_labels'].append(label)

            # Save the grid if required
            if save:
                # Reshape the grid to (channels, height, width) for PyTorch
                grid_with_features = np.transpose(grid_with_features, (2, 0, 1))

                # Create a subdirectory for each grid size (small, medium, large)
                scale_dir = os.path.join(save_dir, size_label)
                os.makedirs(scale_dir, exist_ok=True)

                # Save the grid in the appropriate subdirectory
                grid_filename = os.path.join(scale_dir, f"grid_{i}_{size_label}_class_{int(label)}.npy")
                np.save(grid_filename, grid_with_features)
                print(f"Saved {size_label} grid for point {i} to {grid_filename}")

    return labeled_grids_dict






