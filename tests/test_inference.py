import unittest
import torch
import numpy as np
from scripts.inference import inference
from utils.point_cloud_data_utils import read_las_file_to_numpy
from utils.train_data_utils import remap_labels


class TestInferenceWithDummyModel(unittest.TestCase):

    def setUp(self):
        # Create a dummy model with a simple forward method
        class DummyModel(torch.nn.Module):
            def forward(self, small_grids, medium_grids, large_grids):
                batch_size = small_grids.size(0)
                return torch.randn(batch_size, 5)  # Simulate output for 5 classes

        self.model = DummyModel()
        self.device = torch.device('cpu')  # Use CPU for the dummy model

        # Load actual point cloud data
        data_array, feature_names = read_las_file_to_numpy('data/raw/labeled_FSL.las')

        remapped_array, _ = remap_labels(data_array)
        # Use a small sample of the data for testing
        self.point_cloud_array = remapped_array[np.random.choice(data_array.shape[0], 200, replace=False)]

        # Define parameters for grid generation
        self.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        self.grid_resolution = 128
        self.channels = 10

    def test_inference_with_dummy_model(self):
        # Run inference with grid generation using the dummy model and real data
        predicted_labels = inference(
            model=self.model,
            point_cloud_array=self.point_cloud_array,
            window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution,
            channels=self.channels,
            device=self.device
        )

        # Check that predictions have the correct shape and type
        self.assertIsInstance(predicted_labels, torch.Tensor)
        self.assertEqual(predicted_labels.shape[0], 200)  # Expecting 200 predictions for 200 input points
        print(f"Predicted Labels: {predicted_labels}")


