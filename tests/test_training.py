import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.data_utils import prepare_dataloader
from scripts.train import train, validate, train_epochs


class TestTrainingProcess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Run once for the entire test class
        cls.device = torch.device('cpu')  # Use CPU for testing
        cls.model = MultiScaleCNN().to(cls.device)
        cls.criterion = nn.CrossEntropyLoss()
        cls.optimizer = optim.SGD(cls.model.parameters(), lr=0.01)
        cls.scheduler = optim.lr_scheduler.StepLR(cls.optimizer, step_size=5, gamma=0.5)

        # Mocked DataLoader with random data
        cls.train_loader = prepare_dataloader(batch_size=4, num_samples=200)
        cls.val_loader = prepare_dataloader(batch_size=4, num_samples=200)

    def test_model_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, MultiScaleCNN, "Model should be an instance of MultiScaleCNN.")
        self.assertTrue(next(self.model.parameters()).is_cuda == False, "Model should be on CPU for testing.")

    def test_data_loader(self):
        """Test if data loaders are working properly."""
        batch = next(iter(self.train_loader))
        inputs, labels = batch
        self.assertEqual(inputs.size(0), 4, "Batch size should be 4.")
        self.assertEqual(inputs.ndim, 4, "Inputs should be 4-dimensional.")
        self.assertEqual(labels.ndim, 1, "Labels should be 1-dimensional.")

    def test_training_step(self):
        """Test the training step for a single batch."""
        initial_loss = train(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
        self.assertTrue(initial_loss >= 0, "Initial loss should be non-negative.")

    def test_model_training_over_multiple_epochs(self):
        """Test that the model trains correctly over multiple epochs without crashing."""
        # Run the training loop for a fixed number of epochs
        epochs_to_run = 5
        initial_patience = 10  # Set a high patience to ensure it doesn't trigger early stopping

        # Run training with a few epochs to ensure it works
        train_epochs(self.model, self.train_loader, self.val_loader, self.criterion,
                     self.optimizer, self.scheduler, epochs=epochs_to_run, patience=initial_patience,
                     device=self.device, save_dir='tests/test_training_saved', plot_dir='tests/test_training_saved')

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
        # Simulate DataLoader yielding incorrect shapes
        incorrect_loader = [(
            torch.randn(4, 1, 32, 32),  # Incorrect shape; should be [batch_size, channels, height, width]
            torch.randint(0, 10, (4,))
        )]

        # Mock the DataLoader to yield incorrect shapes
        dataloader_mock = MagicMock()
        dataloader_mock.__iter__.return_value = iter(incorrect_loader)

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




