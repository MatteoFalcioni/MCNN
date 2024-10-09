import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import os
import numpy as np
from utils.point_cloud_data_utils import read_file_to_numpy
from scripts.point_cloud_to_image import generate_multiscale_grids, load_features_used, load_saved_grids
from datetime import datetime
import pandas as pd
import torch.nn as nn


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class GridDataset(Dataset):
    def __init__(self, grids_dict, labels):
        """
        Initializes the dataset by storing the file paths to the grids and loading the provided labels.

        Args:
        - grids_dict (dict): Dictionary containing file paths to the grids for each scale.
        - labels (list): List of labels corresponding to the grid files.
        """
        self.grids_dict = grids_dict
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves a grid and its corresponding label based on the index.

        Args:
        - idx (int): The index of the data point to retrieve.

        Returns:
        - small_grid, medium_grid, large_grid (torch.Tensor): Grids loaded lazily from .npy files.
        - label (torch.Tensor): Corresponding label for the grid.
        """
        # Load grids lazily using the file paths stored in grids_dict
        small_grid_path = self.grids_dict['small'][idx]
        medium_grid_path = self.grids_dict['medium'][idx]
        large_grid_path = self.grids_dict['large'][idx]

        small_grid = torch.tensor(np.load(small_grid_path), dtype=torch.float32)
        medium_grid = torch.tensor(np.load(medium_grid_path), dtype=torch.float32)
        large_grid = torch.tensor(np.load(large_grid_path), dtype=torch.float32)

        # Get the corresponding label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return small_grid, medium_grid, large_grid, label


def prepare_dataloader(batch_size, pre_process_data, data_dir='data/raw/labeled_FSL.las', grid_save_dir='data/pre_processed_data',
                       window_sizes=None, grid_resolution=128, features_to_use=None, train_split=0.8, features_file_path=None):
    """
    Prepares the DataLoader by either preprocessing the data and saving the grids to disk or by loading saved grids.
    Args:
    - batch_size (int): The batch size to be used for training.
    - pre_process_data (bool): If True, generates new grids and saves them to disk.
    - data_dir (str): Path to the raw data (e.g., .las or .csv file).
    - grid_save_dir (str): Directory where the pre-processed grids are saved.
    - window_sizes (list): List of window sizes to use for grid generation (needed if pre_process_data=True).
    - grid_resolution (int): Resolution of the grid (e.g., 128x128).
    - features_to_use (list): List of feature names to use for grid generation.
    - train_split (float): Ratio of the data to use for training (e.g., 0.8 for 80% training data).
    - features_file_path: filepath to feature used in .npy data. Only needed if using raw data in .npy format for pre-processing.

    Returns:
    - train_loader (DataLoader): DataLoader for training.
    - eval_loader (DataLoader): DataLoader for validation (if train_split > 0).
    """

    if pre_process_data:
        # Preprocess and generate grids, always saving them to disk
        if not window_sizes:
            raise ValueError("Window sizes must be provided when generating grids.")

        data_array, known_features = read_file_to_numpy(data_dir=data_dir, features_to_use=features_to_use, features_file_path=features_file_path)

        # Generate and save the grids
        generate_multiscale_grids(data_array, window_sizes, grid_resolution, features_to_use, known_features, grid_save_dir)

    # Load saved grids based on the saved file paths
    print("Loading saved grids...")
    grids_dict, labels = load_saved_grids(grid_save_dir)

    # Create the dataset using the file paths and labels
    full_dataset = GridDataset(grids_dict, labels)

    if train_split > 0.0:
        # Split the dataset into training and evaluation sets
        train_size = int(train_split * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

        # Create DataLoaders for training and evaluation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    else:
        # If no train/test split, create one DataLoader for the full dataset
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = None

    return train_loader, eval_loader



def save_model(model, save_dir='models/saved'):
    """
    Saves the PyTorch model with a filename that includes the current date and time.

    Args:
    - model (nn.Module): The trained model to be saved.
    - save_dir (str): The directory where the model will be saved. Default is 'models/saved'.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get the current date and time
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create the model filename
    model_filename = f"mcnn_model_{current_time}.pth"
    model_save_path = os.path.join(save_dir, model_filename)

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')






