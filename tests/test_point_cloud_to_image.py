import unittest
import numpy as np
from utils.point_cloud_data_utils import read_las_file_to_numpy, numpy_to_dataframe
from scripts.point_cloud_to_image import create_feature_grid, assign_features_to_grid, generate_multiscale_grids
from utils.plot_utils import visualize_grid, visualize_grid_with_comparison
from scipy.spatial import KDTree
import os


class TestPointCloudToImage(unittest.TestCase):

    def setUp(self):
        self.las_file_path = 'data/raw/features_F.las'
        self.sample_size = 10  # Subset for testing. Choosing a very small subset of data to avoid computational overload
        self.grid_resolution = 128
        self.channels = 10
        self.window_size = 20.0

        # Load LAS file, get the data and feature names
        self.full_data, self.feature_names = read_las_file_to_numpy(self.las_file_path)
        self.df = numpy_to_dataframe(self.full_data)
        np.random.seed(42)  # For reproducibility
        self.sampled_data = self.full_data[np.random.choice(self.full_data.shape[0], self.sample_size, replace=False)]

        # Define the window sizes for multiscale grids
        self.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        # Add dummy class labels (e.g., 0 or 1) to unlabeled data
        dummy_class_labels = np.random.randint(0, 2, size=(self.sample_size, 1))
        self.sampled_data_with_class_labels = np.hstack((self.sampled_data, dummy_class_labels))

        self.save_imgs_bool = False  # if True, save the generated images
        self.save_imgs_dir = 'tests/test_feature_imgs'   # directory to save test images
        os.makedirs(self.save_imgs_dir, exist_ok=True)

        self.save_grids_dir = 'tests/test_feature_imgs/test_grid_np'  # directory to save test grids
        os.makedirs(self.save_imgs_dir, exist_ok=True)

    def test_create_and_assign_grids(self):

        # Load the KDTree once for the entire point cloud
        points = self.full_data[:, :2]  # Only x, y coordinates
        tree = KDTree(points)

        # Check that sampled data is not empty and has the expected structure
        self.assertIsInstance(self.full_data, np.ndarray)
        self.assertGreaterEqual(self.sampled_data.shape[1], 4)  # At least x, y, z, and one feature

        # Select a center point
        center_point = self.full_data[100000, :3]

        # Create a grid around the center point
        grid, _, x_coords, y_coords, _ = create_feature_grid(
            center_point, window_size=self.window_size, grid_resolution=self.grid_resolution, channels=self.channels
        )

        # Ensure grid has the correct shape
        self.assertEqual(grid.shape, (self.grid_resolution, self.grid_resolution, self.channels))

         # Assign features using the pre-built KDTree
        grid_with_features = assign_features_to_grid(self.full_data, tree, grid, x_coords, y_coords, channels=self.channels)

        # Ensure features are assigned (grid should not be all zeros)
        self.assertFalse(np.all(grid_with_features == 0), "Grid is unexpectedly empty or all zeros.")

        # Check a few random grid cells to ensure they have diverse values
        print("Sample assigned features in grid:")
        for _ in range(5):  # Check 5 random grid cells
            i, j = np.random.randint(0, self.grid_resolution, 2)
            print(f"Grid cell ({i}, {j}) features: {grid_with_features[i, j, :]}")
            self.assertFalse(np.all(grid_with_features[i, j, :] == 0), "Grid cell features are unexpectedly all zeros.")

        # Visualize and eventually save feature images (if save.bool = True)
        for chan in range(0, self.channels):
            # Create a filename for saving the image
            feature_name = self.feature_names[3 + chan] if len(self.feature_names) > 3 + chan else f"Channel_{chan}"
            file_path = os.path.join(self.save_imgs_dir, f"Grid_Visual_window{int(self.window_size)}_{feature_name}.png")

            # Visualize and save the grid image
            visualize_grid(grid_with_features, channel=chan, title=f"Grid Visualization for {feature_name}", save=self.save_imgs_bool, file_path=file_path)

        # visualize and save feature image compared with point cloud
        chosen_chan = 8  # channel to visualize on feature image (8=nir)
        visualize_grid_with_comparison(grid, self.df, center_point, window_size=self.window_size, feature_names=self.feature_names,
                                       channel=chosen_chan, visual_size=50, save=self.save_imgs_bool, file_path=file_path)

    def test_generate_multiscale_grids(self):
        """
        Test the generation of multiscale grids with associated labels, using a sampled dataset with dummy labels.
        """

        grids_dict = generate_multiscale_grids(self.sampled_data_with_class_labels, self.window_sizes,
                                               self.grid_resolution, self.channels, save_dir=self.save_grids_dir,
                                               save=False)

        # Verify structure of the returned dictionary
        for scale_info in self.window_sizes:
            scale_label = scale_info[0]  # Extract the scale label (small, medium, large)
            self.assertIn(scale_label, grids_dict, f"{scale_label} scale not found in the returned dictionary.")

            # Check that grids and class labels are NumPy arrays
            self.assertIsInstance(grids_dict[scale_label]['grids'], np.ndarray,
                                  f"'grids' is not a NumPy array for {scale_label}.")
            self.assertIsInstance(grids_dict[scale_label]['class_labels'], np.ndarray,
                                  f"'class_labels' is not a NumPy array for {scale_label}.")

            # Ensure grids and class labels are the correct length
            num_grids = grids_dict[scale_label]['grids'].shape[0]
            num_class_labels = grids_dict[scale_label]['class_labels'].shape[0]
            self.assertEqual(num_grids, num_class_labels,
                             f"Number of grids does not match number of class labels for {scale_label} scale.")

            # Validate the grid shapes and class label consistency
            for i in range(num_grids):
                grid = grids_dict[scale_label]['grids'][i]
                class_label = grids_dict[scale_label]['class_labels'][i]

                # Ensure each grid has the correct shape (channels should be the first dimension)
                self.assertEqual(grid.shape, (self.channels, self.grid_resolution, self.grid_resolution),
                                 f"Grid shape is incorrect for {scale_label} scale at index {i}.")

                # Check that the class label is consistent across all scales (small, medium, large)
                for other_scale in self.window_sizes:
                    other_scale_label = other_scale[0]
                    self.assertEqual(grids_dict[other_scale_label]['class_labels'][i], class_label,
                                     f"Class label mismatch between {scale_label} and {other_scale_label} at index {i}.")

    def test_saving_of_grids(self):
        """
        Test that the generated grids are saved correctly with appropriate filenames and shapes.
        """
        # Generate and save the multiscale grids
        grids_dict = generate_multiscale_grids(self.sampled_data_with_class_labels, self.window_sizes,
                                               self.grid_resolution, self.channels, save_dir=self.save_grids_dir,
                                               save=True)

        # Check that the files are saved with the correct names and structure
        for i in range(len(self.sampled_data_with_class_labels)):
            for scale_label, _ in self.window_sizes:
                # Check the filename format
                grid_filename = os.path.join(self.save_grids_dir,
                                             f"grid_{i}_{scale_label}_class_{int(grids_dict[scale_label]['class_labels'][i])}.npy")

                # Ensure the file was saved
                self.assertTrue(os.path.exists(grid_filename), f"{grid_filename} was not saved.")

                # Load the saved grid and check its shape
                saved_grid = np.load(grid_filename)
                self.assertEqual(saved_grid.shape, (self.grid_resolution, self.grid_resolution, self.channels),
                                 f"Grid shape for {grid_filename} is not as expected.")

