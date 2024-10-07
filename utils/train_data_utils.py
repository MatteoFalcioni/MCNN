import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import os
import numpy as np
from utils.point_cloud_data_utils import read_las_file_to_numpy
from scripts.point_cloud_to_image import generate_multiscale_grids
from datetime import datetime
import pandas as pd
import gc

class GridDataset(Dataset):
    def __init__(self, grids_dict, scale, labels):
        self.grids = grids_dict[scale]['grids']
        self.labels = labels

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        small_grid, label = self.small_dataset[idx]
        medium_grid, _ = self.medium_dataset[idx]  # Label is the same, so we ignore it here
        large_grid, _ = self.large_dataset[idx]    # Label is the same, so we ignore it here
        return (small_grid, medium_grid, large_grid, label)


class CombinedGridDataset(Dataset):
    def __init__(self, small_dataset, medium_dataset, large_dataset):
        self.small_dataset = small_dataset
        self.medium_dataset = medium_dataset
        self.large_dataset = large_dataset

    def __len__(self):
        # All datasets should have the same length
        return len(self.small_dataset)

    def __getitem__(self, idx):
        # Retrieve the grids and labels from each dataset
        small_grid, label = self.small_dataset[idx]
        medium_grid, _ = self.medium_dataset[idx]  # Label is the same, so we ignore it here
        large_grid, _ = self.large_dataset[idx]    # Label is the same, so we ignore it here
        return (small_grid, medium_grid, large_grid, label)


def prepare_dataloader(batch_size, pre_process_data, data_dir='data/raw/labeled_FSL.las', grid_save_dir='data/pre_processed_data',
                       window_sizes=None, grid_resolution=128, features_to_use=None, save_grids=True, train_split=0.8):
    labels = []

    if not os.listdir(grid_save_dir) and not pre_process_data:
        raise FileNotFoundError(f"No saved grids found in {grid_save_dir}. Please generate grids first or check the directory.")

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

        for scale in ['small', 'medium', 'large']:
            for file_name in os.listdir(os.path.join(grid_save_dir, scale)):
                grid = np.load(os.path.join(grid_save_dir, scale, file_name))
                grids_dict[scale]['grids'].append(grid)

                if scale == 'small':
                    label = int(file_name.split('_')[-1].split('.')[0].replace('class_', ''))
                    labels.append(label)

    # Create individual datasets for each scale
    small_dataset = GridDataset(grids_dict, 'small', labels)
    medium_dataset = GridDataset(grids_dict, 'medium', labels)
    large_dataset = GridDataset(grids_dict, 'large', labels)

    # Combine the datasets into a single dataset
    full_dataset = CombinedGridDataset(small_dataset, medium_dataset, large_dataset)

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




