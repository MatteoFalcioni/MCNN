import unittest
import numpy as np
import os
from utils.plot_utils import visualize_grid, visualize_grid_with_comparison
from utils.point_cloud_data_utils import read_las_file_to_numpy, remap_labels, numpy_to_dataframe
from scripts.point_cloud_to_image import create_feature_grid, assign_features_to_grid
from scipy.spatial import KDTree

class TestPlotUtils(unittest.TestCase):

    def setUp(self):
        # Path to the directory containing the saved grids
        self.grid_dir = 'tests/test_grids/small'  # Adjust the scale if needed (small, medium, large)
        
        # Load one of the saved grids for testing (adjust the filename if necessary)
        grid_file = os.path.join(self.grid_dir, 'grid_0_small_class_0.npy')
        self.grid_channels_first = np.load(grid_file)  # Assuming the grid is saved in channel-first format

        # Load LAS file, get the data and feature names
        self.sample_size = 1000
        self.las_file_path = 'data/raw/labeled_FSL.las'
        self.full_data, self.feature_names = read_las_file_to_numpy(self.las_file_path)
        self.df = numpy_to_dataframe(data_array=self.full_data, feature_names=self.feature_names)

        # Select a center point
        self.center_point = self.full_data[100, :3]

        # Create a grid around the center point
        self.grid_resolution = 128
        self.channels = 10
        self.window_size = 10.0

        self.grid, _, x_coords, y_coords, _ = create_feature_grid(
            self.center_point, window_size=self.window_size, grid_resolution=self.grid_resolution, channels=self.channels
        )

        # Load the KDTree once for the entire point cloud
        points = self.full_data[:, :2]  # Only x, y coordinates
        tree = KDTree(points)
        # Assign features using the pre-built KDTree
        self.grid = assign_features_to_grid(tree, self.full_data, self.grid, x_coords, y_coords, channels=self.channels)


    def test_visualize_grid(self):
        """
        Test for the visualize_grid function, ensures it can handle channel-first format.
        """
        # The function expects the grid in channel-first format (e.g., [channels, height, width])
        visualize_grid(self.grid_channels_first, channel=5, feature_names=self.feature_names, save=False)  # Test visualization for channel 0

        # We are testing visualization; it's difficult to assert correctness in automated tests,
        # but if there is no exception, we can assume it's functioning.
        print("visualize_grid executed successfully for channel-first grid.")

    def test_visualize_grid_with_comparison(self):
        """
        test for the visualize_grid_with_comparison function
        """
        visualize_grid_with_comparison(self.grid, self.df, self.center_point, window_size=self.window_size, feature_names=self.feature_names,
                                       channel=8, visual_size=50, save=False)
