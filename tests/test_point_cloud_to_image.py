import unittest
import numpy as np
from utils.point_cloud_data_utils import read_las_file_to_numpy, numpy_to_dataframe, read_csv_file_to_numpy, sample_data, load_saved_grids
from scripts.point_cloud_to_image import create_feature_grid, assign_features_to_grid, generate_multiscale_grids, compute_point_cloud_bounds 
from utils.plot_utils import visualize_grid, visualize_grid_with_comparison
from scipy.spatial import cKDTree as KDTree
import os
import pandas as pd

class TestPointCloudToImage(unittest.TestCase):

    def setUp(self):
        self.file_path = 'data/raw/features_F.las'
        # self.file_path = 'data/training_data/test_21.csv'
        self.sample_size = 2000  # Subset for testing. 
        self.grid_resolution = 128
        self.features_to_use = ['intensity', 'red', 'green', 'blue']  # Example selected features
        self.channels = len(self.features_to_use)  # Number of channels based on selected features
        self.window_size = 5.0

        # Load LAS file and get data with user-selected features
        self.full_data, self.feature_names = read_las_file_to_numpy(self.file_path, features_to_extract=self.features_to_use)
        print(f'feature names in csv test file: {self.feature_names}')
        self.df = numpy_to_dataframe(self.full_data, self.feature_names)
        num_points = self.full_data.shape[0]
        labels = np.random.randint(0, 5, size=num_points)
        # Append labels as a new column
        self.data_with_labels = np.hstack((self.full_data, labels.reshape(-1, 1)))
        # np.random.seed(42)  # For reproducibility
        self.sampled_data = self.data_with_labels[np.random.choice(self.full_data.shape[0], self.sample_size, replace=False)]
        self.idx = 100000

        # Define the window sizes for multiscale grids
        self.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]

        self.save_imgs_bool = False  # if True, save the generated images
        self.save_imgs_dir = 'tests/test_feature_imgs'   # directory to save test images
        os.makedirs(self.save_imgs_dir, exist_ok=True)

        self.grid_save_dir = 'tests/multiscale_grids'
        os.makedirs(self.grid_save_dir, exist_ok=True)
        
        # Compute the point cloud bounds
        self.point_cloud_bounds = compute_point_cloud_bounds(self.full_data)

    def test_create_and_assign_grids(self):

        # Load the KDTree once for the entire point cloud
        points = self.full_data[:, :3]  # Use x, y, z coordinates
        tree = KDTree(points)

        # Check that sampled data is not empty and has the expected structure
        self.assertIsInstance(self.full_data, np.ndarray)
        self.assertGreaterEqual(self.full_data.shape[1], 4)  # At least x, y, z, and one feature

        # Select a center point
        center_point = self.full_data[self.idx, :3]

        # Compute the point cloud bounds
        point_cloud_bounds = compute_point_cloud_bounds(self.full_data)

        # Check if the center point is within the point cloud bounds
        half_window = self.window_size / 2
        if (center_point[0] - half_window < self.point_cloud_bounds['x_min'] or 
            center_point[0] + half_window > self.point_cloud_bounds['x_max'] or
            center_point[1] - half_window < self.point_cloud_bounds['y_min'] or
            center_point[1] + half_window > self.point_cloud_bounds['y_max']):
            self.skipTest(f"Skipping test: center point {center_point} is out of bounds for the selected window size.")
    

        # Create a grid around the center point
        grid, _, x_coords, y_coords, z_coord = create_feature_grid(
            center_point, window_size=self.window_size, grid_resolution=self.grid_resolution, channels=self.channels
        )

        # Ensure grid has the correct shape
        self.assertEqual(grid.shape, (self.grid_resolution, self.grid_resolution, self.channels))

        # Identify feature indices dynamically
        print(f'features to use: {self.features_to_use}, known features: {self.feature_names}')
        feature_indices = [self.feature_names.index(feature) for feature in self.features_to_use]
        print(f'feature indices: {feature_indices}')

        # Assign features using the pre-built KDTree
        grid_with_features = assign_features_to_grid(tree, self.full_data, grid, x_coords, y_coords, z_coord, feature_indices)

        # Check how many grid cells are still zero after assigning features
        non_zero_cells = np.count_nonzero(grid_with_features)
        total_cells = grid_with_features.size
        zero_cells_percentage = (total_cells - non_zero_cells) / total_cells * 100

        print(f"Percentage of zero cells in the grid: {zero_cells_percentage}%")# Check how many grid cells are still zero after assigning features
        non_zero_cells = np.count_nonzero(grid_with_features)
        total_cells = grid_with_features.size
        zero_cells_percentage = (total_cells - non_zero_cells) / total_cells * 100

        print(f"Percentage of zero cells in the grid: {zero_cells_percentage}%")


        # Ensure features are assigned (grid should not be all zeros)
        self.assertFalse(np.all(grid_with_features == 0), "Grid is unexpectedly empty or all zeros.")

        # Check that no NaN or Inf values exist in the assigned grid
        self.assertFalse(np.isnan(grid_with_features).any(), "NaN values found in grid features.")
        self.assertFalse(np.isinf(grid_with_features).any(), "Inf values found in grid features.")

        # Check if KDTree queries returned valid indices
        self.assertFalse(np.any(grid_with_features == 0), "Assigned grid cells are unexpectedly empty.")

        # Check a few random grid cells to ensure they have diverse values
        print("Sample assigned features in grid:")
        for _ in range(5):  # Check 5 random grid cells
            i, j = np.random.randint(0, self.grid_resolution, 2)
            self.assertFalse(np.all(grid_with_features[i, j, :] == 0), "Grid cell features are unexpectedly all zeros.")

        # Transpose the grid to match PyTorch's 'channels x height x width' format for visualization
        grid_with_features = np.transpose(grid_with_features, (2, 0, 1))

        """# Visualize and eventually save feature images (if save = True)
        for chan in range(0, self.channels):
            # Create a filename for saving the image
            feature_name = self.feature_names[3 + chan] if len(self.feature_names) > 3 + chan else f"Channel_{chan}"
            file_path = os.path.join(self.save_imgs_dir, f"Grid_Visual_window{int(self.window_size)}_{feature_name}.png")

            # Visualize and save the grid image
            print(f"Grid shape to be visualized: {grid_with_features.shape}")
            visualize_grid(grid_with_features, channel=chan, title=f"Grid Visualization for {feature_name}", save=self.save_imgs_bool, file_path=file_path)

        # visualize and eventually save feature image compared with point cloud
        chosen_chan = 3  # channel to visualize on feature image (8=nir)
        visualize_grid_with_comparison(grid_with_features, self.df, center_point, window_size=self.window_size, feature_names=self.feature_names,
                                       channel=chosen_chan, visual_size=50, save=self.save_imgs_bool, file_path=file_path)
        """

    def test_kd_tree(self):
        """
        Test that the KDTree is constructed properly and that KDTree queries return valid indices.
        """
        points = self.full_data[:, :3]  # Use x, y, z coordinates
        tree = KDTree(points)

        # Verify the KDTree contains all points
        self.assertEqual(tree.n, len(points), "KDTree does not contain all points from the dataset.")

        # Select a center point for querying
        center_point = self.full_data[self.idx, :3]

        # Check if the center point is within the point cloud bounds
        half_window = self.window_size / 2
        if (center_point[0] - half_window < self.point_cloud_bounds['x_min'] or 
            center_point[0] + half_window > self.point_cloud_bounds['x_max'] or
            center_point[1] - half_window < self.point_cloud_bounds['y_min'] or
            center_point[1] + half_window > self.point_cloud_bounds['y_max']):
            self.skipTest(f"Skipping test: center point {center_point} is out of bounds for the selected window size.")

        # Create grid and query KDTree for nearest neighbors
        grid, _, x_coords, y_coords, z_coord = create_feature_grid(
            center_point, window_size=self.window_size, grid_resolution=self.grid_resolution, channels=self.channels
        )

        # Flatten grid coordinates to query the KDTree
        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')
        grid_coords = np.stack((grid_x.flatten(), grid_y.flatten(), np.full(grid_x.size, z_coord)), axis=-1)

        # Query KDTree for nearest points
        distances, indices = tree.query(grid_coords)

        # Ensure no NaN or Inf values are returned by the KDTree query
        self.assertFalse(np.isnan(indices).any(), "KDTree query returned NaN indices.")
        self.assertFalse(np.isinf(indices).any(), "KDTree query returned Inf indices.")

        # Check that the queried indices are within valid range
        self.assertTrue(np.all(indices >= 0), "KDTree query returned invalid (negative) indices.")
        self.assertTrue(np.all(indices < len(self.full_data)), "KDTree query returned out-of-bounds indices.")

        # Check for edge cases (e.g., near the edges of the grid)
        edge_x = grid_x[0, 0]
        edge_y = grid_y[0, 0]
        edge_coord = np.array([edge_x, edge_y, z_coord])
        _, edge_index = tree.query(edge_coord)

        self.assertGreaterEqual(edge_index, 0, "KDTree returned invalid index for edge case.")
        self.assertLess(edge_index, len(self.full_data), "KDTree returned out-of-bounds index for edge case.")

        # Ensure valid distances (no NaN/Inf in distances)
        self.assertFalse(np.isnan(distances).any(), "KDTree returned NaN distances.")
        self.assertFalse(np.isinf(distances).any(), "KDTree returned Inf distances.")

        print("KDTree query results are valid.")
    
    
    def test_generate_multiscale_grids(self):

            print('testing multiscale gen...')
            generate_multiscale_grids(
                data_array=self.sampled_data,
                window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution,
                features_to_use=self.features_to_use,
                known_features=self.feature_names,
                save_dir='tests/multiscale_grids'
            )

            # Verify that the grids and corresponding labels were saved correctly
            for scale_label, _ in self.window_sizes:
                grid_dir = f'tests/multiscale_grids/{scale_label}'
                self.assertTrue(os.path.exists(grid_dir), f"Directory {grid_dir} does not exist.")
                saved_grids = [f for f in os.listdir(grid_dir) if f.endswith('.npy')]
                self.assertGreater(len(saved_grids), 0, f"No grids saved for scale {scale_label}.")

                # Load the grids for each scale and ensure they have valid values
                for grid_file in saved_grids:
                    grid_path = os.path.join(grid_dir, grid_file)
                    grid = np.load(grid_path)

                    # Check for NaN, Inf, or all-zero grids
                    self.assertFalse(np.isnan(grid).any(), f"Grid {grid_file} contains NaN values.")
                    self.assertFalse(np.isinf(grid).any(), f"Grid {grid_file} contains Inf values.")
                    self.assertGreater(np.count_nonzero(grid), 0, f"Grid {grid_file} is all zeros.")

    
    def test_load_saved_grids(self):
        """
        Test loading saved multiscale grids, ensuring consistency across scales and validation of grid contents.
        """
        print("Testing loading of saved grids...")

        # Load the grids and labels
        grids_dict, labels = load_saved_grids(self.grid_save_dir)

        # Ensure grids for all scales are consistent in number
        num_small = len(grids_dict['small'])
        num_medium = len(grids_dict['medium'])
        num_large = len(grids_dict['large'])
        num_labels = len(labels)

        # Check that the number of grids is the same across all scales and labels
        self.assertEqual(num_small, num_medium, "Mismatch between small and medium grid counts.")
        self.assertEqual(num_small, num_large, "Mismatch between small and large grid counts.")
        self.assertEqual(num_small, num_labels, "Mismatch between grid count and label count.")

        print(f"Loaded {num_small} grids for each scale with {num_labels} labels.")

        # Check that the grid paths are valid files
        for scale in ['small', 'medium', 'large']:
            for grid_path in grids_dict[scale]:
                self.assertTrue(os.path.exists(grid_path), f"Grid file not found: {grid_path}")

        # Optionally print the first few grids and labels for manual inspection
        print(f"First 5 grid paths for 'small' scale: {grids_dict['small'][:5]}")
        print(f"First 5 labels: {labels[:5]}")