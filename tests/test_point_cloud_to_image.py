import unittest
import numpy as np
from utils.point_cloud_data_utils import read_las_file_to_numpy
from data.transforms.point_cloud_to_image import create_feature_grid, assign_features_to_grid, generate_grids_for_training
from utils.plot_utils import visualize_grid, plot_point_cloud


class TestPointCloudToImageProcessing(unittest.TestCase):

    def setUp(self):
        self.las_file_path = 'data/raw/features_F.las'
        self.sample_size = 300000  # Subset for testing
        self.grid_resolution = 128
        self.channels = 10
        self.window_size = 10.0

        # Load LAS file, get the data and feature names
        self.full_data, self.feature_names = read_las_file_to_numpy(self.las_file_path)
        np.random.seed(42)  # For reproducibility
        self.sampled_data = self.full_data[np.random.choice(self.full_data.shape[0], self.sample_size, replace=False)]

    def test_create_and_assign_grids(self):

        plot_point_cloud(self.sampled_data)
        # Check that sampled data is not empty and has the expected structure
        self.assertIsInstance(self.sampled_data, np.ndarray)
        self.assertGreaterEqual(self.sampled_data.shape[1], 4)  # At least x, y, z, and one feature

        # Select a center point (could be the first point for simplicity)
        center_point = self.sampled_data[10000, :3]

        # Create a grid around the center point
        grid, cell_size, x_coords, y_coords, z_coords = create_feature_grid(
            center_point, window_size=self.window_size, grid_resolution=self.grid_resolution, channels=self.channels
        )

        # Ensure grid has the correct shape
        self.assertEqual(grid.shape, (self.grid_resolution, self.grid_resolution, self.channels))

        # Assign features to grid
        grid_with_features = assign_features_to_grid(self.sampled_data, grid, x_coords, y_coords, channels=self.channels)

        # Ensure features are assigned (grid should not be all zeros)
        self.assertFalse(np.all(grid_with_features == 0), "Grid is unexpectedly empty or all zeros.")

        # Check a few random grid cells to ensure they have diverse values
        print("Sample assigned features in grid:")
        for _ in range(5):  # Check 5 random grid cells
            i, j = np.random.randint(0, self.grid_resolution, 2)
            print(f"Grid cell ({i}, {j}) features: {grid_with_features[i, j, :]}")
            self.assertFalse(np.all(grid_with_features[i, j, :] == 0), "Grid cell features are unexpectedly all zeros.")

        # Visualize one of the channels to verify visually
        for chan in range(0, self.channels):
            visualize_grid(grid_with_features, channel=chan, title=f"Grid Visualization for {self.feature_names[3+chan]}")

    def test_generate_grids_for_training(self):
        # Generate grids for the sampled dataset
        self.sampled_data = self.full_data[np.random.choice(self.full_data.shape[0], 1000, replace=False)]    # choose a very small subset of data to avoid computational overload
        grids = generate_grids_for_training(self.sampled_data, self.window_size, self.grid_resolution, self.channels)

        # Check if the output is a list
        self.assertIsInstance(grids, list, "The output is not a list.")

        # Check that the number of generated grids matches the number of points
        self.assertEqual(len(grids), len(self.sampled_data),
                         "The number of grids does not match the number of points.")

        # Validate the shape of each grid
        for grid in grids:
            self.assertEqual(grid.shape, (self.grid_resolution, self.grid_resolution, self.channels), "Grid shape is not as expected.")

