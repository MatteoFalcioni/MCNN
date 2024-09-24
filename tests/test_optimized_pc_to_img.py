import unittest
import torch
import os
from utils.point_cloud_data_utils import read_las_file_to_numpy, remap_labels
from scripts.optimized_pc_to_img import gpu_create_feature_grid, gpu_assign_features_to_grid, prepare_grids_dataloader, gpu_generate_multiscale_grids
import numpy as np
import random
from utils.plot_utils import visualize_grid


class TestGPUGridBatchingFunctions(unittest.TestCase):

    def setUp(self):
        # Common parameters for testing
        self.batch_size = 10
        self.grid_resolution = 128
        self.channels = 3
        self.window_size = 10.0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Define window sizes for multiscale grid generation
        self.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        self.save_dir = 'tests/test_optimized_grids/'    # dir to dave generated grids
        os.makedirs(self.save_dir, exist_ok=True)

        self.las_file_path = 'data/raw/labeled_FSL.las'     # Path to the LAS file
        self.sample_size = 1000  # Number of points to sample for the test
        self.full_data, self.feature_names = read_las_file_to_numpy(self.las_file_path)     # Load LAS file, get the data and feature names
        # Random sampling from the full dataset for testing
        np.random.seed(42)  # For reproducibility
        self.sampled_data = self.full_data[np.random.choice(self.full_data.shape[0], self.sample_size, replace=False)]
        self.sampled_data, _ = remap_labels(self.sampled_data)
        self.save_dir_real_data = 'tests/test_optimized_grids_real_data/'
        os.makedirs(self.save_dir_real_data, exist_ok=True)

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

    def test_gpu_create_feature_grid(self):
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
        self.assertEqual(batch_data.shape, (self.batch_size, 2))   # (batch_size, 2 for (x,y))
        self.assertEqual(batch_features.shape, (self.batch_size, self.channels))  # (batch_size, channels)
        self.assertEqual(batch_labels.shape, (self.batch_size, ))     # (batch,size, )

        self.assertEqual(batch_data.device, self.device)
        self.assertEqual(batch_features.device, self.device)
        self.assertEqual(batch_labels.device, self.device)

    def test_assign_features_with_real_data(self):
        """Test that features are correctly assigned to grids with real data."""
        # Prepare the DataLoader
        data_loader = prepare_grids_dataloader(self.sampled_data, self.channels, batch_size=50, num_workers=4)

        # Process the DataLoader batches
        for batch_idx, (batch_data, batch_features, batch_labels) in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}/{len(data_loader)}...")

            # Move data to the correct device (GPU or CPU)
            batch_data = batch_data.to(self.device)
            batch_features = batch_features.to(self.device)

            # Create grids
            grids, _, x_coords, y_coords = gpu_create_feature_grid(batch_data, self.window_size, self.grid_resolution,
                                                                   self.channels, self.device)

            # Assign features to the grids
            updated_grids = gpu_assign_features_to_grid(batch_data, batch_features, grids, x_coords, y_coords,
                                                        self.sampled_data, self.channels, self.device)

            # Check that features have been assigned (e.g., grid values are not all zeros)
            self.assertFalse(torch.all(updated_grids == 0),
                             "Features were not correctly assigned. All grid values are zero.")

            # Check that the grids are not identical (check for variability)
            grid_differences = torch.mean(updated_grids,
                                          dim=(2, 3))  # Mean over the height and width to compare channels
            self.assertFalse(torch.all(grid_differences[0] == grid_differences),
                             "All grids are identical. No feature variability found.")

            # Optionally, print or log values for further analysis
            print(f"Feature assignment successful for batch {batch_idx}.")

    def test_generate_and_save_multiscale_grids_with_real_data(self):
        """Test multiscale grid generation with a real dataset sample."""
        # Prepare the DataLoader with sampled data
        data_loader = prepare_grids_dataloader(self.sampled_data, self.channels, batch_size=50, num_workers=10)

        # Generate and save multiscale grids
        grids_dict = gpu_generate_multiscale_grids(data_loader, self.window_sizes, self.grid_resolution, self.channels, self.device,
                                  full_data=self.sampled_data, save=True, save_dir=self.save_dir_real_data)

        # Check if grids are generated for each scale
        for size_label, _ in self.window_sizes:
            self.assertIn(size_label, grids_dict, f"{size_label} grids are missing from the generated grids.")

            grids = grids_dict[size_label]['grids']
            self.assertGreater(len(grids), 0, f"No {size_label} grids generated.")

        for size_label, _ in self.window_sizes:
            scale_dir = os.path.join(self.save_dir_real_data, size_label)
            self.assertTrue(os.path.exists(scale_dir), f"Directory {scale_dir} does not exist.")

            # Load saved .npy files from the directory
            saved_files = [f for f in os.listdir(scale_dir) if f.endswith('.npy')]
            self.assertGreater(len(saved_files), 0, f"No grids found in directory {scale_dir}.")

            # Randomly select a few grids to visualize
            selected_files = random.sample(saved_files, min(3, len(saved_files)))  # Choose up to 3 files
            for file in selected_files:
                grid_filepath = os.path.join(scale_dir, file)

                # Load the grid from the .npy file
                loaded_grid = np.load(grid_filepath)

                # Visualize the grid
                print(f"Visualizing {file}")
                channel = 2
                feature_name = self.feature_names[channel]
                visualize_grid(loaded_grid, channel=2, title=f"Visualization of {file}", save=False, file_path=file)
