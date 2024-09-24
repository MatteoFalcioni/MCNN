import unittest
import torch
import os
from utils.point_cloud_data_utils import read_las_file_to_numpy
from scripts.optimized_pc_to_img import gpu_generate_multiscale_grids, gpu_create_feature_grid, gpu_assign_features_to_grid


class TestGPUGridFunctions(unittest.TestCase):

    def setUp(self):
        # Setup common parameters for testing
        self.center_point = (10.0, 10.0, 0.0)
        self.window_size = 2.5
        self.grid_resolution = 128
        self.channels = 3
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Simulated point cloud data: 10 points with (x, y, z) and 3 feature values
        self.point_cloud = torch.tensor([
            [9.0, 9.0, 0.0, 0.5, 0.3, 0.2],
            [11.0, 9.0, 0.0, 0.7, 0.8, 0.9],
            [10.0, 11.0, 0.0, 0.2, 0.4, 0.6],
            [9.5, 10.5, 0.0, 0.9, 0.1, 0.3],
            [10.2, 10.2, 0.0, 0.3, 0.5, 0.7],
            [10.8, 10.0, 0.0, 0.4, 0.6, 0.8],
            [9.7, 10.3, 0.0, 0.1, 0.9, 0.5],
            [10.5, 9.6, 0.0, 0.6, 0.2, 0.4],
            [9.9, 10.1, 0.0, 0.8, 0.7, 0.3],
            [10.4, 9.8, 0.0, 0.2, 0.4, 0.9]
        ], device=self.device)

    def test_gpu_create_feature_grid(self):
        """Test that the grid and coordinates are correctly generated on the GPU."""
        grid, cell_size, x_coords, y_coords, z_coords = gpu_create_feature_grid(
            self.center_point, self.window_size, self.grid_resolution, self.channels, device=self.device
        )

        # Assert that grid and coordinates are on the correct device
        self.assertEqual(grid.device, self.device)
        self.assertEqual(x_coords.device, self.device)
        self.assertEqual(y_coords.device, self.device)
        self.assertEqual(z_coords.device, self.device)

        # Check grid shape
        self.assertEqual(grid.shape, (self.grid_resolution, self.grid_resolution, self.channels))
        self.assertEqual(x_coords.shape, (self.grid_resolution,))
        self.assertEqual(y_coords.shape, (self.grid_resolution,))

        # Check that cell size is correct
        expected_cell_size = self.window_size / self.grid_resolution
        self.assertAlmostEqual(cell_size, expected_cell_size)

    def test_gpu_assign_features_to_grid(self):
        """Test that features are assigned correctly to the grid on the GPU."""
        grid, _, x_coords, y_coords, _ = gpu_create_feature_grid(
            self.center_point, self.window_size, self.grid_resolution, self.channels, device=self.device
        )

        # Call feature assignment function
        updated_grid = gpu_assign_features_to_grid(
            self.point_cloud, grid, x_coords, y_coords, self.channels, device=self.device
        )

        # Assert the grid is still on the correct device
        self.assertEqual(updated_grid.device, self.device)

        # Verify that features are being assigned correctly
        # For simplicity, check that the grid is no longer all zeros after assignment
        self.assertFalse(torch.all(updated_grid == 0))

        # Check if the shape remains correct after assignment
        self.assertEqual(updated_grid.shape, (self.grid_resolution, self.grid_resolution, self.channels))
