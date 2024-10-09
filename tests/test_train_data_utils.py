import unittest
from utils.train_data_utils import GridDataset, load_saved_grids

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

