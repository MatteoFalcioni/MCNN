import unittest
import numpy as np
import torch
from scripts.optimized_pc_to_img import gpu_generate_multiscale_grids, prepare_grids_dataloader
from utils.point_cloud_data_utils import read_las_file_to_numpy, remap_labels
from scipy.spatial import KDTree
from utils.plot_utils import visualize_grid

class TestGridFeatureAssignment(unittest.TestCase):
    def setUp(self):
        # Setup for both fake and real data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grid_resolution = 128
        self.window_sizes = [('small', 5.0), ('medium', 10.0), ('large', 20.0)]
        self.channels = 13  # Number of channels to use in the tests

        # Create fake data for testing
        self.fake_data = np.random.rand(1000, 16)  # 1000 points, 16 features (x, y, z + 13 feature channels)
        self.fake_data[:, :3] *= 100  # Scale coordinates for variability

        # Read and prepare real data
        las_file_path = 'data/raw/features_F.las'
        full_data, _ = read_las_file_to_numpy(las_file_path)
        self.real_data = full_data

    def test_fake_data_feature_assignment(self):
        # Create DataLoader for fake data
        fake_data_loader = prepare_grids_dataloader(self.fake_data, batch_size=5, num_workers=0)

        # Run grid generation on fake data with stop_after_batches=1
        labeled_grids_dict = gpu_generate_multiscale_grids(fake_data_loader, self.window_sizes, self.grid_resolution, self.channels, self.device, self.fake_data, stop_after_batches=1)

        # Perform assertions to verify feature assignment (e.g., check values at a specific cell)
        for scale_label in self.window_sizes:
            scale = scale_label[0]
            grid_sample = labeled_grids_dict[scale]['grids'][0]  # Get the first grid
            # Example check: values at cell (5, 5) should not be zero
            self.assertNotEqual(np.sum(grid_sample[:, 5, 5]), 0, "Feature assignment failed for fake data.")

    def test_real_data_feature_assignment(self):
        # Add dummy labels to real data for testing
        dummy_labels = np.zeros((self.real_data.shape[0], 1))
        self.real_data = np.hstack((self.real_data, dummy_labels))
        
        # Create DataLoader for real data
        real_data_loader = prepare_grids_dataloader(self.real_data, batch_size=10, num_workers=0)

        # Run grid generation on real data with stop_after_batches=1
        labeled_grids_dict = gpu_generate_multiscale_grids(real_data_loader, self.window_sizes, self.grid_resolution, self.channels, self.device, self.real_data, stop_after_batches=1)

        # Perform assertions to verify feature assignment
        for scale_label in self.window_sizes:
            scale = scale_label[0]
            grid_sample = labeled_grids_dict[scale]['grids'][0]  # Get the first grid

            print(f'grid sample shape: {grid_sample.shape}')

            # Visualize the first channel of the grid
            visualize_grid(grid_sample[8, :, :, :], channel=8, title=f"Real Data - Scale: {scale}")

if __name__ == '__main__':
    unittest.main()
