import time
import unittest
import numpy as np
from utils.point_cloud_data_utils import read_las_file_to_numpy, numpy_to_dataframe
from scripts.point_cloud_to_image import create_feature_grid, assign_features_to_grid, generate_multiscale_grids
from scripts.new_optimization import opt_create_feature_grid, opt_assign_features_to_grid, opt_generate_multiscale_grids
from utils.plot_utils import visualize_grid, visualize_grid_with_comparison
import os


class TestPointCloudToImage(unittest.TestCase):

    def setUp(self):
        self.las_file_path = 'data/raw/labeled_FSL.las'
        self.sample_size = 200
        self.grid_resolution = 128
        self.channels = 10
        self.window_size = 20.0

        # Load LAS file and sample the data
        self.full_data, self.feature_names = read_las_file_to_numpy(self.las_file_path)
        np.random.seed(42)
        self.sampled_data = self.full_data[np.random.choice(self.full_data.shape[0], self.sample_size, replace=False)]

        self.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]

        self.save_imgs_bool = False
        self.save_grids_dir = 'tests/test_feature_imgs/test_grid_np'
        os.makedirs(self.save_grids_dir, exist_ok=True)

    def test_performance_comparison(self):
        """ Test performance difference between the original and optimized grid generation. """

        # Select a center point
        center_point = self.full_data[100000, :3]

        # Time the original function
        start_time = time.time()
        grid_orig, cell_size_orig, x_coords_orig, y_coords_orig = create_feature_grid(
            center_point, window_size=self.window_size, grid_resolution=self.grid_resolution, channels=self.channels
        )
        original_duration = time.time() - start_time
        print(f"Original create_feature_grid execution time: {original_duration:.4f} seconds")

        # Time the optimized function
        start_time = time.time()
        grid_opt, cell_size_opt, x_coords_opt, y_coords_opt = opt_create_feature_grid(
            center_point, window_size=self.window_size, grid_resolution=self.grid_resolution, channels=self.channels
        )
        optimized_duration = time.time() - start_time
        print(f"Optimized create_feature_grid execution time: {optimized_duration:.4f} seconds")

        # Ensure that the results are consistent
        np.testing.assert_almost_equal(cell_size_orig, cell_size_opt, decimal=5, err_msg="Cell size mismatch")
        np.testing.assert_almost_equal(x_coords_orig, x_coords_opt, decimal=5, err_msg="x_coords mismatch")
        np.testing.assert_almost_equal(y_coords_orig, y_coords_opt, decimal=5, err_msg="y_coords mismatch")

        # Compare durations
        print(f"Speedup factor for create_feature_grid: {original_duration / optimized_duration:.2f}x")

    def test_generate_multiscale_grids_performance(self):
        """ Test performance difference for generating multiscale grids. """

        # Time the original multiscale grid generation
        start_time = time.time()
        grids_dict_orig = generate_multiscale_grids(self.sampled_data, self.window_sizes,
                                                    self.grid_resolution, self.channels, save_dir=self.save_grids_dir,
                                                    save=False)
        original_duration = time.time() - start_time
        print(f"Original generate_multiscale_grids execution time: {original_duration:.4f} seconds")

        # Time the optimized multiscale grid generation
        start_time = time.time()
        grids_dict_opt = opt_generate_multiscale_grids(self.sampled_data, self.window_sizes,
                                                       self.grid_resolution, self.channels, save_dir=self.save_grids_dir,
                                                       save=False, device='cuda')
        optimized_duration = time.time() - start_time
        print(f"Optimized generate_multiscale_grids execution time: {optimized_duration:.4f} seconds")

        # Ensure the number of grids and labels match
        for scale_label in ['small', 'medium', 'large']:
            self.assertEqual(grids_dict_orig[scale_label]['grids'].shape, grids_dict_opt[scale_label]['grids'].shape,
                             f"Grid shape mismatch for {scale_label}")
            self.assertEqual(grids_dict_orig[scale_label]['class_labels'].shape,
                             grids_dict_opt[scale_label]['class_labels'].shape, f"Class label shape mismatch for {scale_label}")

        # Compare durations
        print(f"Speedup factor for generate_multiscale_grids: {original_duration / optimized_duration:.2f}x")
