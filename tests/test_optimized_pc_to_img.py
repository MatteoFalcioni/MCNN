import unittest
import torch
import os
from utils.point_cloud_data_utils import read_las_file_to_numpy, remap_labels
from scripts.optimized_pc_to_img import gpu_create_feature_grid, gpu_assign_features_to_grid, prepare_grids_dataloader, gpu_generate_multiscale_grids
import numpy as np
from utils.plot_utils import visualize_grid
from scipy.spatial import KDTree


class TestGPUGridBatchingFunctions(unittest.TestCase):

    def setUp(self):
        
        self.batch_size = 15     # small batches for simulated data
        self.batch_size_real = 100  # batch size for real data

        self.grid_resolution = 128
        self.channels = 10
        self.window_size = 10.0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Define window sizes for multiscale grid generation
        self.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        self.save_dir = 'tests/test_optimized_grids/'    # dir to dave generated grids
        os.makedirs(self.save_dir, exist_ok=True)

        self.las_file_path = 'data/raw/labeled_FSL.las'     # Path to the LAS file
        self.sample_size = 2000  # Number of points to sample for the test
        self.full_data, self.feature_names = read_las_file_to_numpy(self.las_file_path)     # Load LAS file, get the data and feature names
        # Random sampling from the full dataset for testing
        np.random.seed(42)  # For reproducibility
        self.sampled_data = self.full_data[np.random.choice(self.full_data.shape[0], self.sample_size, replace=False)]
        self.sampled_data, _ = remap_labels(self.sampled_data)

        self.save_dir_real_data = 'tests/test_optimized_grids_real_data/'
        os.makedirs(self.save_dir_real_data, exist_ok=True)

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

    def test_gpu_create_feature_grid(self):
        """Test the gpu_create_feature_grid function."""
        grids, cell_size, x_coords, y_coords, _ = gpu_create_feature_grid(
            torch.tensor(self.data_array[:, :3]), self.window_size, self.grid_resolution, self.channels, device=self.device
        )

        # Test the grid shape
        self.assertEqual(grids.shape, (self.batch_size, self.channels, self.grid_resolution, self.grid_resolution))

        # Test the cell size
        expected_cell_size = self.window_size / self.grid_resolution
        self.assertAlmostEqual(cell_size, expected_cell_size)
        # Test the shape of x_coords and y_coords
        self.assertEqual(x_coords.shape, (self.batch_size, self.grid_resolution))
        self.assertEqual(y_coords.shape, (self.batch_size, self.grid_resolution))

    def test_prepare_grids_dataloader(self):
        data_loader = prepare_grids_dataloader(self.data_array, self.batch_size, num_workers=4)

        # check that the dataloader is not empty
        data_iter = iter(data_loader)
        batch_data = next(data_iter)
        
        # Access the batch data; it should be a single tensor with shape (batch_size, 4)
        batch_tensor = batch_data[0].to(self.device)  # Access the first (and only) element as the combined tensor
        
        # Extract coordinates and labels from the batch tensor
        coordinates = batch_tensor[:, :3]  # First 3 columns for (x, y, z)
        labels = batch_tensor[:, 3].long()  # Last column for labels

        # Verify that the data batch has the expected sizes and shapes
        self.assertEqual(coordinates.shape, (self.batch_size, 3))  # (batch_size, 3 for (x,y,z))
        self.assertEqual(labels.shape, (self.batch_size, ))         # (batch_size, )

        # Check if they are on the correct device
        self.assertEqual(coordinates.device, self.device)
        self.assertEqual(labels.device, self.device)


