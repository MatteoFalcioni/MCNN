import unittest
import torch
import os
import numpy as np
from utils.point_cloud_data_utils import read_las_file_to_numpy
from scripts.optimized_pc_to_img import gpu_generate_multiscale_grids


class TestOptimizedGridGeneration(unittest.TestCase):
    def setUp(self):
        """ Set up the test with the necessary data """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labeled_filepath = 'data/raw/labeled_FSL.las'
        self.data_array, self.features_names = read_las_file_to_numpy(self.labeled_filepath)
        self.sampled_array = self.data_array[np.random.choice(self.data_array.shape[0], 400, replace=False)]
        print(f'features extracted: {self.features_names}')
        self.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        self.grid_resolution = 128
        self.channels = 5
        self.save_dir = 'tests/test_feature_imgs/test_optimized_grids'
        self.save = True

    def test_gpu_grid_generation(self):
        """ Test grid generation on the GPU """
        print("Testing GPU grid generation...")
        grids_dict = gpu_generate_multiscale_grids(
            data_array=self.sampled_array,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            channels=self.channels,
            device=self.device,
            save_dir=self.save_dir,
            save=False  # testing saving later on
        )

        # Check that the grids are generated correctly
        self.assertEqual(len(grids_dict['small']['grids']), 400)
        self.assertEqual(grids_dict['small']['grids'].shape[1:],
                         (self.channels, self.grid_resolution, self.grid_resolution))

    def test_grid_saving(self):
        """ Test that grids are saved correctly """
        print("Testing grid saving functionality...")
        gpu_generate_multiscale_grids(
            data_array=self.sampled_array,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            channels=self.channels,
            device=self.device,
            save_dir=self.save_dir,
            save=self.save
        )

        # Check that the grids have been saved as .npy files
        for size_label, _ in self.window_sizes:
            scale_dir = os.path.join(self.save_dir, size_label)
            self.assertTrue(os.path.exists(scale_dir))

            # Check that there are saved files
            saved_files = os.listdir(scale_dir)
            self.assertGreater(len(saved_files), 0)
            for file_name in saved_files:
                self.assertTrue(file_name.endswith('.npy'))

    def test_parallel_processing(self):
        """ Test that the data is processed in batches """
        print("Testing batch processing (parallel)...")
        batch_size = 100
        for i in range(0, len(self.sampled_array), batch_size):
            batch = self.sampled_array[i:i + batch_size]
            grids_dict = gpu_generate_multiscale_grids(
                data_array=batch,
                window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution,
                channels=self.channels,
                device=self.device,
                save_dir=self.save_dir,
                save=False
            )

            self.assertEqual(len(grids_dict['small']['grids']), min(batch_size, len(self.sampled_array) - i))
