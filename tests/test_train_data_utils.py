import unittest
from utils.train_data_utils import GridDataset, load_saved_grids
from utils.train_data_utils import prepare_dataloader
from torch.utils.data import DataLoader

class TestGridDataset(unittest.TestCase):
    def setUp(self):
        # Load the grids and labels using the load_saved_grids function
        self.grid_save_dir = 'tests/multiscale_grids'  # replace with the actual path
        self.grids_dict, self.labels = load_saved_grids(self.grid_save_dir)

        # Create the dataset instance
        self.dataset = GridDataset(grids_dict=self.grids_dict, labels=self.labels)  

    def test_len(self):
        # Test that the length of the dataset matches the number of grids and labels
        self.assertEqual(len(self.dataset), len(self.labels))

    def test_getitem(self):
        # Test retrieving a sample grid and label
        sample_idx = 0 
        small_grid, medium_grid, large_grid, label = self.dataset[sample_idx]
        
        # Test that the grids are non-empty
        self.assertIsNotNone(small_grid)
        self.assertIsNotNone(medium_grid)
        self.assertIsNotNone(large_grid)
        
        # Check that the retrieved label matches the corresponding one
        self.assertEqual(label, self.labels[sample_idx])


class TestPrepareDataloader(unittest.TestCase):

    def setUp(self):
        # Configurable parameters for the test
        self.batch_size = 16
        self.grid_save_dir = 'tests/multiscale_grids'
        self.pre_process_data = False  # Since we're focusing on loading saved grids
        self.grid_resolution = 128
        # self.data_dir = 'data/raw/labeled_FSL.las'  # Path to raw data (not needed for this test)
        # self.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        # self.features_to_use = ['intensity', 'red', 'green', 'blue']

    def test_dataloader_full_dataset(self):
        # Load the full dataset using the prepare_dataloader function
        train_loader, eval_loader = prepare_dataloader(
            batch_size=self.batch_size,
            pre_process_data=self.pre_process_data,
            # data_dir=self.data_dir,
            grid_save_dir=self.grid_save_dir,
            # window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            # features_to_use=self.features_to_use,
            train_split=0.0  # No train/test split, using full dataset
        )

        # Ensure that the DataLoader is returned properly and not None
        self.assertIsInstance(train_loader, DataLoader, "train_loader is not a DataLoader instance.")
        self.assertIsNone(eval_loader, "eval_loader should be None when train_split=0.0.")

        # Fetch the first batch and verify its structure
        first_batch = next(iter(train_loader))
        self.assertEqual(len(first_batch), 4, "Expected 4 elements in the batch (small_grid, medium_grid, large_grid, label).")
        
        small_grid, medium_grid, large_grid, labels = first_batch
        self.assertEqual(small_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Small grid resolution mismatch.")
        self.assertEqual(medium_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Medium grid resolution mismatch.")
        self.assertEqual(large_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Large grid resolution mismatch.")
        
        # Print some of the first batch data to check if it's loaded correctly
        print("Small Grid Shape:", small_grid.shape)
        print("Medium Grid Shape:", medium_grid.shape)
        print("Large Grid Shape:", large_grid.shape)
        print("Labels:", labels)

