import numpy as np


def create_feature_grid(center_point, window_size=10.0, grid_resolution=128, channels = 3):
    """
    Creates a grid around the center point and initializes cells to store feature values.

    Args:
    - center_point (tuple): The (x, y, z) coordinates of the center point of the grid.
    - window_size (float): The size of the square window around the center point (in meters).
    - grid_resolution (int): The number of cells in one dimension of the grid (e.g., 128 for a 128x128 grid).

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
    grid = np.zeros((grid_resolution, grid_resolution, channels))  # 3 channels for RGB encoding

    # Generate cell coordinates for the grid based on the center point
    i_indices = np.arange(grid_resolution)
    j_indices = np.arange(grid_resolution)

    x_coords = center_point[0] - (64.5 - j_indices) * cell_size
    y_coords = center_point[1] - (64.5 - i_indices) * cell_size
    z_coords = np.full((grid_resolution, grid_resolution), center_point[2])  # Z coordinate is constant for all cells

    return grid, cell_size, x_coords, y_coords, z_coords




