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
        
        batch_data = batch_data.to(self.device)

        # split data and data to the correct device
        coordinates = batch_data[:, :2]
        labels = batch_data[:, 2]
        
        # verify that the data batch has the expected sizes and shapes
        self.assertEqual(coordinates.shape, (self.batch_size, 2))   # (batch_size, 2 for (x,y))
        self.assertEqual(labels.shape, (self.batch_size, ))     # (batch,size, )

        self.assertEqual(coordinates.device, self.device)
        self.assertEqual(labels.device, self.device)

    def test_assign_features_with_real_data(self):
        """Test that features are correctly assigned to grids with real data."""
        # Prepare the DataLoader
        data_loader = prepare_grids_dataloader(self.sampled_data, self.batch_size_real, num_workers=4)
        
        # Load the KDTree once for the entire point cloud
        points = self.sampled_data[:, :3]  # Use x, y, z coordinates for KDTree
        tree = KDTree(points)

        # Process the DataLoader batches
        for batch_idx, batch_data in enumerate(data_loader):

            coordinates = batch_data[0].to(self.device)  # Access the first element as the tensor
            labels = batch_data[1].to(self.device)        # Access the second element for labels 

            # Create grids
            grids, _, x_coords, y_coords, constant_z = gpu_create_feature_grid(coordinates, self.window_size, self.grid_resolution, self.channels, self.device)

            # Assign features to the grids
            updated_grids = gpu_assign_features_to_grid(coordinates, grids, x_coords, y_coords, constant_z, self.sampled_data, tree, self.channels, self.device)
            # Check that features have been assigned (e.g., grid values are not all zeros)
            self.assertFalse(torch.all(updated_grids == 0),
                             "Features were not correctly assigned. All grid values are zero.")

            # Check that the grids are not identical (check for variability)
            grid_differences = torch.mean(updated_grids,
                                          dim=(2, 3))  # Mean over the height and width to compare channels
            self.assertFalse(torch.all(grid_differences[0] == grid_differences),
                             "All grids are identical. No feature variability found.")

            for grid_idx in range(grids.shape[0]):  # Loop through the batch size
                grid = updated_grids[grid_idx]
                class_label = labels[grid_idx].item()

                # Ensure the grid contains non-zero values (features are assigned)
                self.assertFalse(np.all(grid.cpu().numpy() == 0), f"Grid {grid_idx} for batch {batch_idx} is empty. No features assigned.")

                # Pick a random cell to verify feature assignment
                row, col = np.random.randint(0, self.grid_resolution, size=2)
                feature_values = grid[:, row, col]

                # Calculate the cell center coordinates
                cell_center_coords = np.array([x_coords[grid_idx][col].item(), y_coords[grid_idx][row].item(), constant_z[grid_idx].item()])

                # Calculate distances using the full 3D coordinates
                distances = np.linalg.norm(self.sampled_data[:, :3] - cell_center_coords, axis=1)
                nearest_point_idx = np.argmin(distances)
                expected_features = self.sampled_data[nearest_point_idx, 3:3 + self.channels]

                # Check if the assigned features match the nearest point's features
                np.testing.assert_almost_equal(
                    feature_values.cpu().numpy(), 
                    expected_features, 
                    decimal=4,
                    err_msg=f"Feature mismatch for grid {grid_idx} at cell ({row}, {col}) in batch {batch_idx}."
                )

    def test_generate_multiscale_grids_with_real_data(self):
        """Test multiscale grid generation with real data."""
        # Prepare the DataLoader
        data_loader = prepare_grids_dataloader(self.sampled_data, self.batch_size_real, num_workers=4)
        # Generate multiscale grids
        labeled_grids_dict = gpu_generate_multiscale_grids(data_loader, self.window_sizes, self.grid_resolution, self.channels, self.device, full_data=self.sampled_data, save=False)


        # Check if grids are generated for each scale
        for size_label, _ in self.window_sizes:
            self.assertIn(size_label, labeled_grids_dict, f"{size_label} grids are missing from the generated grids.")
            grids = labeled_grids_dict[size_label]['grids']
            self.assertGreater(len(grids), 0, f"No {size_label} grids generated.")
            self.assertEqual(grids[0].shape,
                             (self.batch_size_real, self.channels, self.grid_resolution, self.grid_resolution))


    def test_save_and_load_grids_with_real_data(self):
        """Test saving and loading of grids generated with real data."""
        data_loader = prepare_grids_dataloader(self.sampled_data, batch_size=self.batch_size_real, num_workers=4)
        gpu_generate_multiscale_grids(data_loader, self.window_sizes, self.grid_resolution, self.channels, self.device, full_data=self.sampled_data, save_dir=self.save_dir_real_data, save=True)

        # Verify the saved grids exist
        for size_label, _ in self.window_sizes:
            scale_dir = os.path.join(self.save_dir_real_data, size_label)
            self.assertTrue(os.path.exists(scale_dir), f"{size_label} directory does not exist.")
            saved_files = [f for f in os.listdir(scale_dir) if f.endswith('.npy')]
            self.assertGreater(len(saved_files), 0, f"No grids found in {scale_dir}.")

            # Load one of the saved grids and visualize
            grid_filename = os.path.join(scale_dir, saved_files[0])
            loaded_grid = np.load(grid_filename)
            self.assertEqual(loaded_grid.shape, (self.channels, self.grid_resolution, self.grid_resolution),
                             f"Loaded grid shape is incorrect for {size_label}.")

            # Visualize the loaded grid for verification
            print(f"Visualizing {saved_files[0]}")
            visualize_grid(loaded_grid, channel=7, title=f"Visualization of {saved_files[0]}", save=False)
