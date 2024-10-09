import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.point_cloud_data_utils import extract_num_channels, extract_num_classes
from utils.train_data_utils import prepare_dataloader, initialize_weights
from scripts.train import train, validate, train_epochs


class TestTrainingProcess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Run once for the entire test class
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cls.grid_save_dir = 'tests/multiscale_grids'
        cls.num_channels = extract_num_channels(preprocessed_data_dir=cls.grid_save_dir)
        cls.num_classes = extract_num_classes(pre_process_data=False, preprocessed_data_dir=cls.grid_save_dir)
        cls.model = MultiScaleCNN(channels=cls.num_channels, classes=cls.num_classes).to(cls.device)
        cls.model.apply(initialize_weights)
        cls.criterion = nn.CrossEntropyLoss()
        cls.optimizer = optim.SGD(cls.model.parameters(), lr=0.01)
        cls.scheduler = optim.lr_scheduler.StepLR(cls.optimizer, step_size=5, gamma=0.5)

        # Mocked DataLoader with random data
        cls.train_loader, cls.val_loader = prepare_dataloader(
                                                                batch_size=16,
                                                                pre_process_data=False,                                                                    
                                                                grid_save_dir='tests/multiscale_grids',
                                                                grid_resolution=128,
                                                                train_split=0.8  
                                                            )

    def test_model_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, MultiScaleCNN, "Model should be an instance of MultiScaleCNN.")
        self.assertTrue(next(self.model.parameters()).is_cuda == False, "Model should be on CPU for testing.")

    def test_integration_loader_model(self):
        """Test the integration between the DataLoader and the MultiScaleCNN model."""
        # Get a batch from the DataLoader
        first_batch = next(iter(self.train_loader))

        # Extract the grids and labels
        small_grid, medium_grid, large_grid, labels = first_batch

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
        initial_loss = train(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
        self.assertTrue(initial_loss >= 0, "Initial loss should be non-negative.")

    def test_model_training_over_multiple_epochs(self):
        """Test that the model trains correctly over multiple epochs without crashing."""
        # Run the training loop for a fixed number of epochs
        epochs_to_run = 3
        initial_patience = 10  # Set a high patience to ensure it doesn't trigger early stopping

        # Run training with a few epochs to ensure it works
        train_epochs(self.model, self.train_loader, self.val_loader, self.criterion,
                     self.optimizer, self.scheduler, epochs=epochs_to_run, patience=initial_patience,
                     device=self.device, save_dir='tests/test_training_saved', plot_dir='tests/test_training_saved',
                     save=False)

        # Check that the training completes without any exceptions
        self.assertTrue(True, "Model training did not complete as expected.")

    def test_validation_step(self):
        """Test the validation step for a single batch."""
        val_loss = validate(self.model, self.val_loader, self.criterion, self.device)
        self.assertTrue(val_loss >= 0, "Validation loss should be non-negative.")

    def test_empty_data_loader(self):
        """Test training and validation steps with an empty DataLoader."""
        empty_loader = []  # Simulating an empty DataLoader
        with self.assertRaises(StopIteration):
            next(iter(empty_loader))

    def test_incorrect_input_shape(self):
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
            with self.assertRaises(ValueError):  # Adjust this to match your actual error handling
                train(self.model, self.train_loader, mock_criterion, self.optimizer, self.device)

    def test_early_stopping_trigger(self):
        """Test early stopping when validation loss does not improve."""
        # Mock `validate` to always return the same loss, simulating no improvement
        with patch('scripts.train.validate', return_value=1.0) as mock_validate:
            train_epochs(self.model, self.train_loader, self.val_loader, self.criterion,
                         self.optimizer, self.scheduler, epochs=10, patience=2, device=self.device,
                         save_dir='tests/test_training_saved', plot_dir='tests/test_training_saved')

            # Check how many times validate was called
            self.assertLessEqual(mock_validate.call_count, 3, "Early stopping did not trigger correctly.")


    def test_sanity_check_full_pipeline(self):
        """Run a sanity check for the full training pipeline with a small dataset."""
        # Run training loop for 2 epochs as a sanity check
        epochs_to_run = 2

        # Capture initial training and validation loss
        initial_train_loss = train(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
        initial_val_loss = validate(self.model, self.val_loader, self.criterion, self.device)

        # Run the training loop for a few epochs
        train_epochs(self.model, self.train_loader, self.val_loader, self.criterion,
                    self.optimizer, self.scheduler, epochs=epochs_to_run, patience=3,
                    device=self.device, save_dir='tests/sanity_check', plot_dir='tests/sanity_check', save=False)

        # Capture final training and validation loss
        final_train_loss = train(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
        final_val_loss = validate(self.model, self.val_loader, self.criterion, self.device)
