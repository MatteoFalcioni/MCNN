import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import os
import numpy as np
from utils.point_cloud_data_utils import read_las_file_to_numpy
from scripts.point_cloud_to_image import generate_multiscale_grids
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
        Initializes the dataset by storing the file paths to the grids and their corresponding labels.

        Args:
        - grids_dict (dict): Dictionary containing file paths to the grids ('small', 'medium', 'large').
        - labels (list): List of labels for each grid.
        """
        self.grids_dict = grids_dict
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Lazily loads and returns the grids and corresponding label for a given index.

        Args:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - small_grid (torch.Tensor): Loaded small scale grid.
        - medium_grid (torch.Tensor): Loaded medium scale grid.
        - large_grid (torch.Tensor): Loaded large scale grid.
        - label (int): Corresponding label.
        """
        # Lazily load the grids from the saved paths
        small_grid_path = self.grids_dict['small'][idx]
        medium_grid_path = self.grids_dict['medium'][idx]
        large_grid_path = self.grids_dict['large'][idx]

        # Load the grids from disk
        small_grid = np.load(small_grid_path)
        medium_grid = np.load(medium_grid_path)
        large_grid = np.load(large_grid_path)

        # Convert to PyTorch tensors
        small_grid = torch.tensor(small_grid, dtype=torch.float32)
        medium_grid = torch.tensor(medium_grid, dtype=torch.float32)
        large_grid = torch.tensor(large_grid, dtype=torch.float32)

        # Load the label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return small_grid, medium_grid, large_grid, label


def prepare_dataloader(batch_size, pre_process_data, data_dir='data/raw/labeled_FSL.las', grid_save_dir='data/pre_processed_data',
                       window_sizes=None, grid_resolution=128, features_to_use=None, train_split=0.8):
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

    Returns:
    - train_loader (DataLoader): DataLoader for training.
    - eval_loader (DataLoader): DataLoader for validation (if train_split > 0).
    """

    if pre_process_data:
        # Preprocess and generate grids, always saving them to disk
        if not window_sizes:
            raise ValueError("Window sizes must be provided when generating grids.")

        if data_dir.endswith('.npy'):
            raise ValueError("Simple numpy files cannot be used for data preprocessing, feature names info is needed.")
        elif data_dir.endswith('.las'):
            print("Generating new grids from raw LAS data...")
            data_array, known_features = read_las_file_to_numpy(data_dir, features_to_extract=features_to_use)
        elif data_dir.endswith('.csv'):
            print("Generating new grids from raw CSV data...")
            df = pd.read_csv(data_dir)
            data_array = df.values
            known_features = df.columns.tolist()

        # Generate and save the grids
        grids_dict = generate_multiscale_grids(data_array, window_sizes, grid_resolution, features_to_use, known_features, grid_save_dir, save=True)

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


def load_saved_grids(grid_save_dir):
    """
    Loads saved grid file paths and corresponding labels based on common identifiers across 'small', 'medium', and 'large' scales.

    Args:
    - grid_save_dir (str): Directory where the grids are saved.
    
    Returns:
    - grids_dict (dict): Dictionary containing the file paths for each grid.
    - labels (list): List of labels corresponding to the grids.
    """
    grids_dict = {'small': [], 'medium': [], 'large': []}
    labels = []  # Initialize labels list

    # Helper function to extract the common identifier (e.g., 'grid_4998') from the filename
    def get_common_identifier(filename):
        return '_'.join(filename.split('_')[:2])  # Extracts 'grid_4998' from 'grid_4998_small_class_0.npy'

    # Collect filenames and extract identifiers for each scale
    small_files = {get_common_identifier(f): f for f in os.listdir(os.path.join(grid_save_dir, 'small'))}
    medium_files = {get_common_identifier(f): f for f in os.listdir(os.path.join(grid_save_dir, 'medium'))}
    large_files = {get_common_identifier(f): f for f in os.listdir(os.path.join(grid_save_dir, 'large'))}

    # Find common identifiers across all three scales
    common_identifiers = set(small_files.keys()).intersection(medium_files.keys(), large_files.keys())

    if not common_identifiers:
        raise FileNotFoundError("No common grid files found across small, medium, and large scales.")

    # Sort the common identifiers to ensure consistent ordering
    common_identifiers = sorted(common_identifiers)

    # Store file paths for each scale
    for identifier in common_identifiers:
        try:
            small_path = os.path.join(grid_save_dir, 'small', small_files[identifier])
            medium_path = os.path.join(grid_save_dir, 'medium', medium_files[identifier])
            large_path = os.path.join(grid_save_dir, 'large', large_files[identifier])

            grids_dict['small'].append(small_path)
            grids_dict['medium'].append(medium_path)
            grids_dict['large'].append(large_path)

            # Extract the label from the filename (assuming it's the same across scales)
            label = int(small_files[identifier].split('_')[-1].split('.')[0].replace('class_', ''))
            labels.append(label)

        except Exception as e:
            print(f"Error loading files for identifier {identifier}: {e}")
            continue

    print(f"Number of common grids: {len(common_identifiers)}")
    return grids_dict, labels



