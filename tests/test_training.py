import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.point_cloud_data_utils import  extract_num_classes
from utils.train_data_utils import prepare_dataloader
from scripts.train import train, validate, train_epochs


class TestTrainingProcess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Run once for the entire test class
        cls.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cls.full_data_filepath = 'data/datasets/sampled_full_dataset/sampled_data_5251680.csv'
        cls.subset_file = 'data/datasets/train_dataset.csv'
        cls.selected_features = ['intensity', 'red']
        cls.num_channels = len(cls.selected_features)  # Determine the number of channels based on selected features
        cls.num_classes = extract_num_classes(raw_file_path=cls.full_data_filepath) # determine the number of classes from the raw data
        cls.model = MultiScaleCNN(channels=cls.num_channels, classes=cls.num_classes).to(cls.device)
        cls.criterion = nn.CrossEntropyLoss()
        cls.optimizer = optim.SGD(cls.model.parameters(), lr=0.01)
        cls.scheduler = optim.lr_scheduler.StepLR(cls.optimizer, step_size=1, gamma=0.5)
        cls.grid_resolution = 128
        cls.window_sizes = [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
        cls.batch_size = 16
        cls.num_workers = 32

        # Mocked DataLoader with random data
        cls.train_loader, cls.val_loader = prepare_dataloader(
                                                                batch_size=cls.batch_size,
                                                                data_filepath=cls.full_data_filepath,
                                                                window_sizes=cls.window_sizes,
                                                                grid_resolution=cls.grid_resolution,
                                                                features_to_use=cls.selected_features,
                                                                train_split=0.8, 
                                                                num_workers=cls.num_workers,
                                                                subset_file=cls.subset_file
                                                            )


    def test_model_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, MultiScaleCNN, "Model should be an instance of MultiScaleCNN.")
        self.assertTrue(next(self.model.parameters()).is_cuda == True, "Model should be on GPU.")


    def test_integration_loader_model(self):
        """Test the integration between the DataLoader and the MultiScaleCNN model."""
        # Get a batch from the DataLoader
        first_batch = next(iter(self.train_loader))

        # Extract the grids and labels
        small_grid, medium_grid, large_grid, labels, indices = first_batch
        
        # Move the input grids to the same device as the model (GPU in this case)
        small_grid = small_grid.to(self.device)
        medium_grid = medium_grid.to(self.device)
        large_grid = large_grid.to(self.device)
        labels = labels.to(self.device)
        
        # Ensure labels are within valid range
        if labels.min() < 0 or labels.max() >= self.model.classes:
            raise ValueError(f"Labels out of bounds: min {labels.min()}, max {labels.max()}")

        # Ensure that the grids and labels have the correct shapes and types
        self.assertEqual(small_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Small grid resolution mismatch.")
        self.assertEqual(medium_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Medium grid resolution mismatch.")
        self.assertEqual(large_grid.shape[-2:], (self.grid_resolution, self.grid_resolution), "Large grid resolution mismatch.")
        self.assertIsInstance(labels, torch.Tensor, "Labels should be a tensor.")
        self.assertEqual(labels.dtype, torch.long, "Labels should be of type long.")

        # Pass the grids through the model
        outputs = self.model(small_grid, medium_grid, large_grid)

        # Check that the output shape matches the number of classes
        self.assertEqual(outputs.shape[1], self.model.classes, f"Output shape mismatch. Expected {self.model.classes} classes.")

        # Check that the model output is not NaN or Inf
        self.assertFalse(torch.isnan(outputs).any(), "Model output contains NaN values.")
        self.assertFalse(torch.isinf(outputs).any(), "Model output contains Inf values.")


    def test_training_step(self):
        """Test the training step for a single batch."""
        # Get a batch from the DataLoader
        first_batch = next(iter(self.train_loader))

        # Extract the grids and labels
        small_grid, medium_grid, large_grid, labels, _ = first_batch
        small_grid, medium_grid, large_grid, labels = (
            small_grid.to(self.device),
            medium_grid.to(self.device),
            large_grid.to(self.device),
            labels.to(self.device),
        )

        # Perform a forward and backward pass
        outputs = self.model(small_grid, medium_grid, large_grid)
        loss = self.criterion(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Ensure loss is a valid number
        self.assertFalse(torch.isnan(loss).any(), "Loss contains NaN values.")
        self.assertFalse(torch.isinf(loss).any(), "Loss contains Inf values.")


    '''def test_model_training_over_multiple_epochs(self):
        """Test that the model trains correctly over multiple epochs."""
        epochs_to_run = 2
        train_epochs(
            self.model, self.train_loader, self.val_loader, self.criterion,
            self.optimizer, self.scheduler, epochs=epochs_to_run, patience=5,
            device=self.device, save=False
        )

        # Check that the training completes without any exceptions
        self.assertTrue(True, "Model training did not complete as expected.")'''


    def test_validation_step(self):
        """Test the validation step."""
        val_loss = validate(self.model, self.val_loader, self.criterion, self.device)
        self.assertTrue(val_loss >= 0, "Validation loss should be non-negative.")


    def test_empty_data_loader(self):
        """Test training and validation steps with an empty DataLoader."""
        empty_loader = []  # Simulating an empty DataLoader
        with self.assertRaises(StopIteration):
            next(iter(empty_loader))


    '''def test_incorrect_input_shape(self):
        """Test the model with incorrect input shapes to ensure it raises an error."""
        # Simulate DataLoader yielding 4 elements with incorrect shapes
        incorrect_loader = [(
            torch.randn(4, 1, 32, 32),  # Incorrect shape for small_grids
            torch.randn(4, 1, 32, 32),  # Incorrect shape for medium_grids
            torch.randn(4, 1, 32, 32),  # Incorrect shape for large_grids
            torch.randint(0, 10, (4,))  # Labels
        )]

        # Mock the DataLoader to yield the incorrect shapes
        dataloader_mock = MagicMock()
        dataloader_mock.__iter__.return_value = iter(incorrect_loader)

        # Expect a RuntimeError due to the incorrect shape
        with self.assertRaises(RuntimeError):
            train(self.model, dataloader_mock, self.criterion, self.optimizer, self.device)


    def test_training_with_nan_loss(self):
        """Test training with data that will produce NaN loss to check error handling."""

        def mock_criterion(outputs, labels):
            # Simulate a loss that is NaN but still requires gradients
            loss = torch.tensor(float('nan'), requires_grad=True)
            return loss

        with patch('torch.nn.CrossEntropyLoss', mock_criterion):
            with self.assertRaises(ValueError): 
                train(self.model, self.train_loader, mock_criterion, self.optimizer, self.device)


    def test_early_stopping_trigger(self):
        """Test early stopping when validation loss does not improve."""
        # Mock `validate` to always return the same loss, simulating no improvement
        with patch('scripts.train.validate', return_value=1.0) as mock_validate:
            train_epochs(
                self.model, self.train_loader, self.val_loader, self.criterion,
                self.optimizer, self.scheduler, epochs=10, patience=2, device=self.device, save=False
            )

            # Check how many times validate was called (accounting for early stopping)
            self.assertLessEqual(mock_validate.call_count, 3, "Early stopping did not trigger correctly.")'''



    '''def test_sanity_check_full_pipeline(self):
        """Run a sanity check for the full training pipeline with a small dataset."""
        # Run training loop for 2 epochs as a sanity check
        epochs_to_run = 2

        # Capture initial training and validation loss
        initial_train_loss = train(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
        initial_val_loss = validate(self.model, self.val_loader, self.criterion, self.device)

        # Run the training loop for a few epochs
        train_epochs(self.model, self.train_loader, self.val_loader, self.criterion,
                    self.optimizer, self.scheduler, epochs=epochs_to_run, patience=3,
                    device=self.device, model_save_dir='tests/sanity_check/model', save=False)

        # Capture final training and validation loss
        final_train_loss = train(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
        final_val_loss = validate(self.model, self.val_loader, self.criterion, self.device)
        
        print(f'initial train loss: {initial_train_loss}')
        print(f'initial val loss: {initial_val_loss}')
        print(f'final train loss: {final_train_loss}')
        print(f'final val loss: {final_val_loss}')'''
