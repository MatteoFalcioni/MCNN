import unittest
import torch
import os
from utils.point_cloud_data_utils import read_las_file_to_numpy
from scripts.optimized_pc_to_img import gpu_create_feature_grid, gpu_assign_features_to_grid, prepare_grids_dataloader, gpu_generate_multiscale_grids


class TestGPUGridBatchingFunctions(unittest.TestCase):

    def setUp(self):
        # Common parameters for testing
        self.batch_size = 10
        self.grid_resolution = 128
        self.channels = 3
        self.window_size = 2.5
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Simulated batch of center points (x, y, z)
        self.center_points = torch.tensor([
            [10.0, 10.0, 15.0],
            [20.0, 20.0, 8.0],
            [30.0, 30.0, 9.0],
            [40.0, 40.0, 8.0],
            [50.0, 50.0, 22.0],
            [60.0, 60.0, 21.0],
            [70.0, 70.0, 5.0],
            [80.0, 80.0, 5.0],
            [90.0, 90.0, 9.0],
            [100.0, 100.0, 18.0]
        ])

        # Simulated batch data for feature assignment (x, y coordinates)
        self.batch_data = torch.tensor([
            [9.0, 9.0], [11.0, 9.0], [10.0, 11.0], [9.5, 10.5], [10.2, 10.2],
            [19.0, 19.0], [21.0, 19.0], [20.0, 21.0], [19.5, 20.5], [20.2, 20.2]
        ])

        # Simulated features for each point (e.g., RGB features)
        self.batch_features = torch.tensor([
            [0.5, 0.3, 0.2],
            [0.7, 0.8, 0.9],
            [0.2, 0.4, 0.6],
            [0.3, 0.5, 0.7],
            [0.1, 0.2, 0.3],
            [0.6, 0.4, 0.3],
            [0.9, 0.8, 0.7],
            [0.4, 0.3, 0.2],
            [0.2, 0.1, 0.3],
            [0.7, 0.6, 0.8]
        ])

        # Example data_array for DataLoader test (x,y,z + features + labels)
        self.data_array = torch.tensor([
            [10.0, 10.0, 5.0, 0.5, 0.3, 0.2, 1],
            [20.0, 20.0, 6.0, 0.7, 0.8, 0.9, 2],
            [30.0, 30.0, 3.0, 0.2, 0.4, 0.6, 3],
            [10.0, 10.0, 5.0, 0.5, 0.3, 0.2, 1],
            [20.0, 20.0, 6.0, 0.7, 0.8, 0.9, 2],
            [30.0, 30.0, 3.0, 0.2, 0.4, 0.6, 3],
            [10.0, 10.0, 5.0, 0.5, 0.3, 0.2, 1],
            [20.0, 20.0, 6.0, 0.7, 0.8, 0.9, 2],
            [30.0, 30.0, 3.0, 0.2, 0.4, 0.6, 3],
            [10.0, 10.0, 5.0, 0.5, 0.3, 0.2, 1],
            [20.0, 20.0, 6.0, 0.7, 0.8, 0.9, 2],
            [30.0, 30.0, 3.0, 0.2, 0.4, 0.6, 3],
            [10.0, 10.0, 5.0, 0.5, 0.3, 0.2, 1],
            [20.0, 20.0, 6.0, 0.7, 0.8, 0.9, 2],
            [30.0, 30.0, 3.0, 0.2, 0.4, 0.6, 3]
        ]).numpy()

    def test_create_feature_grid(self):
        """Test the gpu_create_feature_grid function."""
        grids, cell_size, x_coords, y_coords = gpu_create_feature_grid(
            self.center_points, self.window_size, self.grid_resolution, self.channels, device=self.device
        )

        # Test the grid shape
        self.assertEqual(grids.shape, (self.batch_size, self.channels, self.grid_resolution, self.grid_resolution))

        # Test the cell size
        expected_cell_size = self.window_size / self.grid_resolution
        self.assertAlmostEqual(cell_size, expected_cell_size)

        # Test the shape of x_coords and y_coords
        self.assertEqual(x_coords.shape, (self.batch_size, self.grid_resolution))
        self.assertEqual(y_coords.shape, (self.batch_size, self.grid_resolution))

    def test_assign_features_to_grid(self):
        """Test the gpu_assign_features_to_grid function."""
        # First, create a batch of grids using the gpu_create_feature_grid function
        grids, cell_size, x_coords, y_coords = gpu_create_feature_grid(
            self.center_points, self.window_size, self.grid_resolution, self.channels, device=self.device
        )

        # Assign features to the grids using gpu_assign_features_to_grid
        updated_grids = gpu_assign_features_to_grid(
            self.batch_data, self.batch_features, grids, x_coords, y_coords, self.channels, device=self.device
        )

        # Ensure the updated grids are on the correct device
        self.assertEqual(updated_grids.device, self.device)

        # Check that the shape of the updated grids remains correct
        self.assertEqual(updated_grids.shape, (self.batch_size, self.channels, self.grid_resolution, self.grid_resolution))

        # Verify that features have been assigned (no zeros in the grid)
        self.assertFalse(torch.all(updated_grids == 0))

    def test_prepare_grids_dataloader(self):
        data_loader = prepare_grids_dataloader(self.data_array, self.channels, self.batch_size, num_workers=4)

        # check that the dataloader is not empty
        data_iter = iter(data_loader)
        batch = next(data_iter)

        batch_data, batch_features, batch_labels = batch

        # Move data to the correct device
        batch_data = batch_data.to(self.device)
        batch_features = batch_features.to(self.device)
        batch_labels = batch_labels.to(self.device)

        # verify that the data batch has the expected sizes and shapes
        self.assertEqual(batch_data.shape, (self.batch_size, 3))   # (batch_size, 3 for (x,y,z))
        self.assertEqual(batch_features.shape, (self.batch_size, self.channels))  # (batch_size, channels)
        self.assertEqual(batch_labels.shape, (self.batch_size, ))     # (batch,size, )

        self.assertEqual(batch_data.device, self.device)
        self.assertEqual(batch_features.device, self.device)
        self.assertEqual(batch_labels.device, self.device)
