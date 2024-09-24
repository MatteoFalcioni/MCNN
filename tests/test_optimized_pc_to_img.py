import unittest
import torch
import os
from utils.point_cloud_data_utils import read_las_file_to_numpy
from scripts.optimized_pc_to_img import gpu_create_feature_grid, gpu_assign_features_to_grid, prepare_grids_dataloader


class TestGPUGridBatchingFunctions(unittest.TestCase):

    def setUp(self):
        # Setup common parameters for testing
        self.batch_size = 3
        self.center_points = torch.tensor([
            [10.0, 10.0, 0.0],
            [20.0, 20.0, 0.0],
            [30.0, 30.0, 0.0]
        ])
        self.window_size = 2.5  # Smaller window size
        self.grid_resolution = 128
        self.channels = 3
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'using device: {self.device}')

        # Simulated batch point cloud data: [batch_size, num_points, (x, y, z)] + features
        self.batch_data = torch.tensor([
            [[9.0, 9.0], [11.0, 9.0], [10.0, 11.0], [9.5, 10.5], [10.2, 10.2]],
            [[19.0, 19.0], [21.0, 19.0], [20.0, 21.0], [19.5, 20.5], [20.2, 20.2]],
            [[29.0, 29.0], [31.0, 29.0], [30.0, 31.0], [29.5, 30.5], [30.2, 30.2]]
        ], device=self.device)

        # Simulated features for the point clouds [batch_size, num_points, channels]
        self.batch_features = torch.tensor([
            [[0.5, 0.3, 0.2], [0.7, 0.8, 0.9], [0.2, 0.4, 0.6], [0.9, 0.1, 0.3], [0.3, 0.5, 0.7]],
            [[0.1, 0.6, 0.2], [0.3, 0.9, 0.5], [0.7, 0.2, 0.4], [0.6, 0.8, 0.1], [0.4, 0.3, 0.9]],
            [[0.2, 0.7, 0.5], [0.9, 0.3, 0.1], [0.8, 0.6, 0.4], [0.3, 0.5, 0.9], [0.6, 0.2, 0.7]]
        ], device=self.device)

        # Example data_array for DataLoader test
        self.data_array = torch.tensor([
            [10.0, 10.0, 0.0, 0.5, 0.3, 0.2, 1],
            [20.0, 20.0, 0.0, 0.7, 0.8, 0.9, 2],
            [30.0, 30.0, 0.0, 0.2, 0.4, 0.6, 3]
        ]).numpy()

    def test_gpu_create_feature_grid_batch(self):
        """Test that grids are created correctly for a batch of center points."""
        grids, cell_size, x_coords, y_coords = gpu_create_feature_grid(
            self.center_points, self.window_size, self.grid_resolution, self.channels, device=self.device
        )

        # Assert that grids and coordinates are on the correct device
        self.assertEqual(grids.device, self.device)
        self.assertEqual(x_coords.device, self.device)
        self.assertEqual(y_coords.device, self.device)

        # Check grid shape for the entire batch
        self.assertEqual(grids.shape, (self.batch_size, self.channels, self.grid_resolution, self.grid_resolution))
        self.assertEqual(x_coords.shape, (self.batch_size, self.grid_resolution))
        self.assertEqual(y_coords.shape, (self.batch_size, self.grid_resolution))

        # Check that the cell size is correct
        expected_cell_size = self.window_size / self.grid_resolution
        self.assertAlmostEqual(cell_size, expected_cell_size)

    def test_gpu_assign_features_to_grid_batch(self):
        """Test that features are assigned correctly for a batch of grids."""
        # Create a batch of grids first
        grids, _, x_coords, y_coords = gpu_create_feature_grid(
            self.center_points, self.window_size, self.grid_resolution, self.channels, device=self.device
        )

        # Call batched feature assignment function
        updated_grids = gpu_assign_features_to_grid(
            self.batch_data, self.batch_features, grids, x_coords, y_coords, self.channels, device=self.device
        )

        # Assert the updated grids are on the correct device
        self.assertEqual(updated_grids.device, self.device)

        # Verify that features have been assigned by checking that the grids are no longer all zeros
        self.assertFalse(torch.all(updated_grids == 0))

        # Check if the shape remains correct after assigning features
        self.assertEqual(updated_grids.shape,
                         (self.batch_size, self.channels, self.grid_resolution, self.grid_resolution))

    def test_feature_assignment_accuracy(self):
        """Test that features are assigned accurately based on nearest points."""
        grids, _, x_coords, y_coords = gpu_create_feature_grid(
            self.center_points, self.window_size, self.grid_resolution, self.channels, device=self.device
        )

        # Manually calculate expected nearest point for the first grid cell (top-left corner) for each batch
        # For simplicity, we're just going to check the top-left corner for each batch

        # Expected nearest points based on the closest to the top-left grid cell in each batch
        # In this case, let's say we're testing the top-left grid cell [0, 0]
        # Example manual expected feature values based on closest points from the batch data
        expected_nearest_point_features = torch.tensor([
            [0.5, 0.3, 0.2],  # Nearest point features for batch 1 (based on manual check)
            [0.1, 0.6, 0.2],  # Nearest point features for batch 2
            [0.2, 0.7, 0.5]  # Nearest point features for batch 3
        ], device=self.device)

        # Call the batched feature assignment function
        updated_grids = gpu_assign_features_to_grid(
            self.batch_data, self.batch_features, grids, x_coords, y_coords, self.channels, device=self.device
        )

        # Check if the first grid cell in each batch (top-left corner) has the correct features
        for batch_idx in range(self.batch_size):
            first_grid_cell_features = updated_grids[batch_idx, :, 0, 0].cpu()  # Extract features for [0, 0] grid cell
            expected_features = expected_nearest_point_features[batch_idx].cpu()  # Expected features for that batch

            # Compare the assigned features with the manually expected features
            self.assertTrue(torch.allclose(first_grid_cell_features, expected_features),
                            f"Features do not match for batch {batch_idx}, cell [0,0].")

    def test_prepare_grids_dataloader(self):
        """Test that DataLoader batches the data correctly."""
        data_loader = prepare_grids_dataloader(self.data_array, self.channels, self.batch_size, num_workers=1,
                                         device=self.device)

        # Check that the DataLoader is not empty and the batches are of the correct size
        for batch_idx, (batch_data, batch_features, batch_labels) in enumerate(data_loader):
            batch_data = batch_data.to(self.device)
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.assertEqual(batch_data.shape, (self.batch_size, 3))  # (batch_size, 3) for (x, y, z)
            self.assertEqual(batch_features.shape, (self.batch_size, self.channels))  # (batch_size, channels)
            self.assertEqual(batch_labels.shape, (self.batch_size,))  # (batch_size,)
