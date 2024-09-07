import unittest
from unittest.mock import patch
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
        cls.train_loader = prepare_dataloader(batch_size=4)
        cls.val_loader = prepare_dataloader(batch_size=4)

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

    def test_validation_step(self):
        """Test the validation step for a single batch."""
        val_loss = validate(self.model, self.val_loader, self.criterion, self.device)
        self.assertTrue(val_loss >= 0, "Validation loss should be non-negative.")

    @patch('utils.plot_utils.plot_loss')
    def test_train_epochs_with_early_stopping(self, mock_plot_loss):
        """Test the full training loop with early stopping and ensure plotting is called."""
        train_epochs(self.model, self.train_loader, self.val_loader, self.criterion,
                     self.optimizer, self.scheduler, epochs=5, device=self.device,
                     save_dir='models/saved/', patience=2)

        mock_plot_loss.assert_called_once()  # Ensure plot_loss is called once at the end

    def test_empty_data_loader(self):
        """Test training and validation steps with an empty DataLoader."""
        empty_loader = []  # Simulating an empty DataLoader
        with self.assertRaises(StopIteration):
            next(iter(empty_loader))

    def test_incorrect_input_shape(self):
        """Test the model with incorrect input shapes to ensure it raises an error."""
        incorrect_loader = [torch.randn(4, 1, 32, 32), torch.randint(0, 10, (4,))]  # Incorrect input shape
        with self.assertRaises(RuntimeError):
            train(self.model, incorrect_loader, self.criterion, self.optimizer, self.device)

    def test_training_with_nan_loss(self):
        """Test training with data that will produce NaN loss to check error handling."""

        def mock_criterion(outputs, labels):
            return torch.tensor(float('nan'))  # Return NaN to simulate an unstable training scenario

        with patch('torch.nn.CrossEntropyLoss', mock_criterion):
            with self.assertRaises(ValueError):
                train(self.model, self.train_loader, mock_criterion, self.optimizer, self.device)

    def test_early_stopping_trigger(self):
        """Test early stopping when validation loss does not improve."""
        with patch('train.validate', return_value=1.0):  # Mock validate to always return a constant loss
            train_epochs(self.model, self.train_loader, self.val_loader, self.criterion,
                         self.optimizer, self.scheduler, epochs=10, device=self.device,
                         save_dir='models/saved/', patience=2)
            self.assertLessEqual(len(self.train_loader), 3, "Early stopping did not trigger correctly.")
