import unittest
import torch
from utils.train_data_utils import prepare_dataloader


class TestPrepareDataLoader(unittest.TestCase):

    def setUp(self):
        """Set up the parameters for testing."""
        self.batch_size = 2  # Small batch size for testing
        self.train_split = 0.8  # 80% training, 20% evaluation
        self.device = torch.device('cpu')

        self.grid_save_dir = 'tests/test_feature_imgs/test_grid_np'

    def test_train_loader(self):
        """Test that the train DataLoader is functioning correctly."""
        # Call the prepare_dataloader function with a small test dataset
        train_loader, eval_loader = prepare_dataloader(
            batch_size=self.batch_size,
            pre_process_data=False,  # Assume the grids are pre-saved
            grid_save_dir=self.grid_save_dir,
            save_grids=False,  # No need to save grids for testing
            train_split=self.train_split
        )

        # Test training set
        for batch_idx, (small_grids, medium_grids, large_grids, labels) in enumerate(train_loader):
            with self.subTest(batch_idx=batch_idx):
                self.assertEqual(small_grids.shape[0], self.batch_size, "Incorrect batch size for small grids")
                self.assertEqual(medium_grids.shape[0], self.batch_size, "Incorrect batch size for medium grids")
                self.assertEqual(large_grids.shape[0], self.batch_size, "Incorrect batch size for large grids")
                self.assertEqual(labels.shape[0], self.batch_size, "Incorrect batch size for labels")
                # Stop after one batch for simplicity
                break

        # Test evaluation set
        for batch_idx, (small_grids, medium_grids, large_grids, labels) in enumerate(eval_loader):
            with self.subTest(batch_idx=batch_idx):
                self.assertEqual(small_grids.shape[0], self.batch_size, "Incorrect batch size for small grids")
                self.assertEqual(medium_grids.shape[0], self.batch_size, "Incorrect batch size for medium grids")
                self.assertEqual(large_grids.shape[0], self.batch_size, "Incorrect batch size for large grids")
                self.assertEqual(labels.shape[0], self.batch_size, "Incorrect batch size for labels")
                # Stop after one batch for simplicity
                break

