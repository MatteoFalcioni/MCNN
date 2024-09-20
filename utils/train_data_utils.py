import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import numpy as np
from utils.point_cloud_data_utils import read_las_file_to_numpy
from scripts.point_cloud_to_image import generate_multiscale_grids
from datetime import datetime


def prepare_dataloader(batch_size, pre_process_data, data_dir='data/raw', grid_save_dir='data/pre_processed_data',
                       window_sizes=None, grid_resolution=None, channels=None, save_grids=True, train_split=0.8):
    """
        Prepares DataLoader objects for training and evaluation with three grid sizes (small, medium, large) and labels.

        If `pre_process_data` is True, the function will generate new grids from raw data. If False, it will load
        pre-saved grids from the specified directory.

        Args:
        - batch_size (int): Size of the batches for training and evaluation.
        - pre_process_data (bool): Whether to generate new grids from raw data or load pre-saved grids.
        - data_dir (str): Directory where raw LiDAR data is stored if generating grids. Default is 'data/raw'.
        - grid_save_dir (str): Directory where the generated grids will be stored or loaded from. Default is 'data/pre_processed_data'.
        - window_sizes (list): List of window sizes for grid generation (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)]).
        - grid_resolution (int): Resolution of the grid (e.g., 128x128). Required if generating grids.
        - channels (int): Number of channels in the grids (e.g., 3). Required if generating grids.
        - save_grids (bool): Whether to save the generated grids to disk. Default is True.
        - train_split (float): Proportion of the data to be used for training (between 0 and 1). Default is 0.8 (80%).

        Returns:
        - train_loader (DataLoader): DataLoader for the training set.
        - eval_loader (DataLoader): DataLoader for the evaluation set.
        """

    small_grids = []
    medium_grids = []
    large_grids = []
    labels = []

    if not os.listdir(grid_save_dir) and not pre_process_data:
        raise FileNotFoundError(
            f"No saved grids found in {grid_save_dir}. Please generate grids first or check the directory.")

    if pre_process_data:
        if not window_sizes or not grid_resolution or not channels:
            raise ValueError("Window sizes, grid resolution, and channels must be provided when generating grids.")

        print("Generating new grids from raw data...")
        data_array = read_las_file_to_numpy(data_dir)
        grids_dict = generate_multiscale_grids(data_array, window_sizes, grid_resolution, channels, grid_save_dir,
                                               save=save_grids)

        # Extract grids for each scale and class labels from grids_dict
        small_grids = torch.stack(grids_dict['small']['grids'])
        medium_grids = torch.stack(grids_dict['medium']['grids'])
        large_grids = torch.stack(grids_dict['large']['grids'])
        labels = torch.tensor(grids_dict['small']['class_labels'], dtype=torch.long)

    else:
        # Load saved grids from the directory
        print("Loading saved grids...")
        for scale in ['small', 'medium', 'large']:
            for file_name in os.listdir(os.path.join(grid_save_dir, scale)):
                grid = np.load(os.path.join(grid_save_dir, scale, file_name))
                if scale == 'small':
                    small_grids.append(torch.tensor(grid, dtype=torch.float32))
                elif scale == 'medium':
                    medium_grids.append(torch.tensor(grid, dtype=torch.float32))
                elif scale == 'large':
                    large_grids.append(torch.tensor(grid, dtype=torch.float32))

                # Extract class label from filename (assuming format like grid_0_small_class_X.npy)
                if scale == 'small':
                    label = int(file_name.split('_')[-1].split('.')[0].replace('class_', ''))
                    labels.append(label)

        small_grids = torch.stack(small_grids)
        medium_grids = torch.stack(medium_grids)
        large_grids = torch.stack(large_grids)
        labels = torch.tensor(labels, dtype=torch.long)

    # Create a TensorDataset with three grid sizes and labels
    dataset = TensorDataset(small_grids, medium_grids, large_grids, labels)

    # Split the dataset into training and evaluation sets
    train_size = int(train_split * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # Create DataLoaders for training and evaluation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

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

