import unittest
from torch.utils.data import DataLoader
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels
from scripts.point_cloud_to_image import compute_point_cloud_bounds
from scipy.spatial import cKDTree
from utils.train_data_utils import PointCloudDataset, prepare_dataloader


class TestPointCloudDataset(unittest.TestCase):
    def setUp(self):
        # Mock point cloud data for the test
        self.data_array, self.known_features = read_file_to_numpy(data_dir='data/sampled/sampled_data_500000.csv')
        self.data_array, _ = remap_labels(self.data_array)
        self.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        self.grid_resolution = 128
        self.features_to_use = ['intensity', 'red', 'green', 'blue']

        # Build the KDTree once for the test
        self.kdtree = cKDTree(self.data_array[:, :3])
        self.point_cloud_bounds = compute_point_cloud_bounds(self.data_array)

        # Create the dataset instance
        self.dataset = PointCloudDataset(
            data_array=self.data_array,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            known_features=self.known_features
        )

    def test_len(self):
        # Test that the length of the dataset matches the number of points
        self.assertEqual(len(self.dataset), len(self.data_array))

    def test_getitem(self):
        # Test retrieving a sample grid and label
        sample_idx = 0
        result = self.dataset[sample_idx]

        # If the point is valid (not skipped), result should not be None
        if result is not None:
            small_grid, medium_grid, large_grid, label = result

            # Test that the grids are non-empty
            self.assertIsNotNone(small_grid)
            self.assertIsNotNone(medium_grid)
            self.assertIsNotNone(large_grid)

            # Check that the retrieved label matches the expected label
            self.assertEqual(label, self.data_array[sample_idx, -1])
        else:
            self.assertIsNone(result, "The point should have been skipped.")


class TestPrepareDataloader(unittest.TestCase):

    def setUp(self):
        # Configurable parameters for the test
        self.batch_size = 16
        self.grid_resolution = 128
        self.train_split = 0.8
        self.data_dir = 'data/sampled/sampled_data_500000.csv'  # Path to raw data for this test
        self.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        self.features_to_use = ['intensity', 'red', 'green', 'blue']

    def test_dataloader_full_dataset(self):
        # Prepare DataLoader without train/test split (using full dataset)
        train_loader, eval_loader = prepare_dataloader(
            batch_size=self.batch_size,
            data_dir=self.data_dir,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            train_split=0.0  # No train/test split, using full dataset
        )

        # Ensure that the train_loader is returned properly and eval_loader is None
        self.assertIsInstance(train_loader, DataLoader, "train_loader is not a DataLoader instance.")
        self.assertIsNone(eval_loader, "eval_loader should be None when train_split=0.0.")

        # Fetch the first batch and verify its structure
        first_batch = next(iter(train_loader))
        self.assertIsNotNone(first_batch, "First batch should not be None (all points skipped).")
        self.assertEqual(len(first_batch), 4, "Expected 4 elements in the batch (small_grid, medium_grid, large_grid, label).")

        small_grid, medium_grid, large_grid, labels = first_batch
        self.assertEqual(small_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Small grid resolution mismatch.")
        self.assertEqual(medium_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Medium grid resolution mismatch.")
        self.assertEqual(large_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Large grid resolution mismatch.")

    def test_train_eval_dataloader(self):
        # Load both train and eval DataLoader
        train_loader, eval_loader = prepare_dataloader(
            batch_size=self.batch_size,
            data_dir=self.data_dir,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            features_to_use=self.features_to_use,
            train_split=self.train_split
        )

        # Test the train_loader
        first_batch = next(iter(train_loader))
        self.assertIsNotNone(first_batch, "First batch should not be None (all points skipped).")
        small_grid, medium_grid, large_grid, labels = first_batch
        self.assertEqual(small_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Small grid resolution mismatch.")
        self.assertEqual(medium_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Medium grid resolution mismatch.")
        self.assertEqual(large_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Large grid resolution mismatch.")
        self.assertLessEqual(len(labels), self.batch_size, "Batch size should be smaller or equal to specified batch size due to skipped points.")

        # Test the eval_loader (ensure it was created and works)
        self.assertIsNotNone(eval_loader, "Evaluation DataLoader is None when it shouldn't be.")
        first_batch_eval = next(iter(eval_loader))
        self.assertIsNotNone(first_batch_eval, "First batch of eval_loader should not be None (all points skipped).")
        small_grid, medium_grid, large_grid, labels_eval = first_batch_eval
        self.assertEqual(small_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Small grid resolution mismatch in eval.")
        self.assertEqual(medium_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Medium grid resolution mismatch in eval.")
        self.assertEqual(large_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Large grid resolution mismatch in eval.")
        self.assertLessEqual(len(labels_eval), self.batch_size, "Batch size should be smaller or equal to specified batch size in eval due to skipped points.")