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
        self.small_grids = grids_dict['small']['grids']
        self.medium_grids = grids_dict['medium']['grids']
        self.large_grids = grids_dict['large']['grids']
        self.labels = labels

    def __len__(self):
        return len(self.small_grids)

    def __getitem__(self, idx):
        # Convert grids to tensors on the fly
        small_grid = torch.tensor(self.small_grids[idx], dtype=torch.float32)
        medium_grid = torch.tensor(self.medium_grids[idx], dtype=torch.float32)
        large_grid = torch.tensor(self.large_grids[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return small_grid, medium_grid, large_grid, label



def prepare_dataloader(batch_size, pre_process_data, data_dir='data/raw/labeled_FSL.las', grid_save_dir='data/pre_processed_data',
                       window_sizes=None, grid_resolution=128, features_to_use=None, save_grids=True, train_split=0.8):
    labels = []
    """os.path.exists(grid_save_dir)

    if not os.listdir(grid_save_dir):
        if not pre_process_data:
            raise FileNotFoundError(f"No saved grids found in {grid_save_dir}. Please generate grids first or check the directory.")
        """
    if pre_process_data:
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

        grids_dict = generate_multiscale_grids(data_array, window_sizes, grid_resolution, features_to_use, known_features, grid_save_dir, save=save_grids)
        labels = grids_dict['small']['class_labels']
    else:
        print("Loading saved grids...")
        grids_dict = {'small': {'grids': []}, 'medium': {'grids': []}, 'large': {'grids': []}}
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

        # Load grids for each scale based on common identifiers
        for identifier in common_identifiers:
            try:
                # Load the grids for each scale using the corresponding filenames
                small_grid = np.load(os.path.join(grid_save_dir, 'small', small_files[identifier]))
                medium_grid = np.load(os.path.join(grid_save_dir, 'medium', medium_files[identifier]))
                large_grid = np.load(os.path.join(grid_save_dir, 'large', large_files[identifier]))
                
                grids_dict['small']['grids'].append(small_grid)
                grids_dict['medium']['grids'].append(medium_grid)
                grids_dict['large']['grids'].append(large_grid)

                # Extract the label from the filename (assuming the label is the same across scales)
                label = int(small_files[identifier].split('_')[-1].split('.')[0].replace('class_', ''))
                labels.append(label)

            except Exception as e:
                print(f"Error loading files for identifier {identifier}: {e}")
                continue

        print(f"Number of common grids: {len(common_identifiers)}")


    # Create the grids dataset
    full_dataset = GridDataset(grids_dict, labels)

    if train_split > 0.0:
        # Split the combined dataset into training and evaluation sets
        train_size = int(train_split * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

        # Create DataLoaders for training and evaluation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    else:
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




