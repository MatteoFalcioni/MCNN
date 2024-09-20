import unittest
import numpy as np
from utils.point_cloud_data_utils import read_las_file_to_numpy, numpy_to_dataframe
from data.transforms.point_cloud_to_image import create_feature_grid, assign_features_to_grid, generate_grids_for_training, generate_multiscale_grids
from utils.plot_utils import visualize_grid, visualize_grid_with_comparison
import os


class TestPointCloudToImage(unittest.TestCase):

    def setUp(self):
        self.las_file_path = 'data/raw/features_F.las'
        self.sample_size = 200  # Subset for testing. Choosing a very small subset of data to avoid computational overload
        self.grid_resolution = 128
        self.channels = 10
        self.window_size = 20.0

        # Load LAS file, get the data and feature names
        self.full_data, self.feature_names = read_las_file_to_numpy(self.las_file_path)
        self.df = numpy_to_dataframe(self.full_data)
        np.random.seed(42)  # For reproducibility
        self.sampled_data = self.full_data[np.random.choice(self.full_data.shape[0], self.sample_size, replace=False)]

        self.save_bool = False  # if True, save the generated images
        self.save_dir = 'tests/test_feature_imgs'   # directory to save test images
        os.makedirs(self.save_dir, exist_ok=True)

    def test_create_and_assign_grids(self):

        # Check that sampled data is not empty and has the expected structure
        self.assertIsInstance(self.full_data, np.ndarray)
        self.assertGreaterEqual(self.sampled_data.shape[1], 4)  # At least x, y, z, and one feature

        # Select a center point
        center_point = self.full_data[100000, :3]

        # Create a grid around the center point
        grid, cell_size, x_coords, y_coords, z_coords = create_feature_grid(
            center_point, window_size=self.window_size, grid_resolution=self.grid_resolution, channels=self.channels
        )

        # Ensure grid has the correct shape
        self.assertEqual(grid.shape, (self.grid_resolution, self.grid_resolution, self.channels))

        # Assign features to grid
        grid_with_features = assign_features_to_grid(self.full_data, grid, x_coords, y_coords, channels=self.channels)

        # Ensure features are assigned (grid should not be all zeros)
        self.assertFalse(np.all(grid_with_features == 0), "Grid is unexpectedly empty or all zeros.")

        # Check a few random grid cells to ensure they have diverse values
        print("Sample assigned features in grid:")
        for _ in range(5):  # Check 5 random grid cells
            i, j = np.random.randint(0, self.grid_resolution, 2)
            print(f"Grid cell ({i}, {j}) features: {grid_with_features[i, j, :]}")
            self.assertFalse(np.all(grid_with_features[i, j, :] == 0), "Grid cell features are unexpectedly all zeros.")

        # Visualize grid's channels to verify visually
        for chan in range(0, self.channels):
            visualize_grid(grid_with_features, channel=chan, title=f"Grid Visualization for {self.feature_names[3+chan]}")

        # Visualize and eventually save feature images (if save.bool = True)
        for chan in range(0, self.channels):
            # Create a filename for saving the image
            feature_name = self.feature_names[3 + chan] if len(self.feature_names) > 3 + chan else f"Channel_{chan}"
            file_path = os.path.join(self.save_dir, f"Grid_Visual_window{int(self.window_size)}_{feature_name}.png")

            # Visualize and save the grid image
            visualize_grid(grid_with_features, channel=chan, title=f"Grid Visualization for {feature_name}", save=self.save_bool, file_path=file_path)

        # visualize and save feature image compared with point cloud
        chosen_chan = 8  # channel to visualize on feature image (8=nir)
        visualize_grid_with_comparison(grid, self.df, center_point, window_size=self.window_size, feature_names=self.feature_names,
                                       channel=chosen_chan, visual_size=50, save=self.save_bool, file_path=file_path)

    def test_generate_grids_for_training(self):
        # Generate grids for the sampled dataset
        grids = generate_grids_for_training(self.sampled_data, self.window_size, self.grid_resolution, self.channels)

        # Check if the output is a list
        self.assertIsInstance(grids, list, "The output is not a list.")

        # Check that the number of generated grids matches the number of points
        self.assertEqual(len(grids), len(self.sampled_data),
                         "The number of grids does not match the number of points.")

        # Validate the shape of each grid
        for grid in grids:
            self.assertEqual(grid.shape, (self.grid_resolution, self.grid_resolution, self.channels), "Grid shape is not as expected.")

    def test_generate_multiscale_grids(self):
        """
        Test the generation and saving of multiscale grids.
        """

        # Define the window sizes
        window_sizes = [('small', 5.0), ('medium', 10.0), ('large', 20.0)]

        generate_multiscale_grids(self.sampled_data, window_sizes, self.grid_resolution, self.channels, 'tests/test_feature_imgs/test_grid_np')

        # Check if files for all scales were saved correctly
        for i in range(len(self.sampled_data)):
            for size_label, _ in window_sizes:
                grid_filename = os.path.join(self.save_dir, f"grid_{i}_{size_label}.npy")

                # Check that the file exists
                self.assertTrue(os.path.exists(grid_filename), f"{grid_filename} was not saved.")

                # Load the saved grid and check its shape
                grid = np.load(grid_filename)
                self.assertEqual(grid.shape, (self.grid_resolution, self.grid_resolution, self.channels),
                                 f"Grid shape for {grid_filename} is not as expected.")

