import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from datetime import datetime
from utils.point_cloud_data_utils import read_las_file_to_numpy
from scripts.point_cloud_to_image import generate_multiscale_grids


def prepare_dataloader(batch_size, generate_grids, data_dir='data/raw', grid_save_dir='data/pre_processed_data', window_sizes=None, grid_resolution=None, channels=None, device=None):
    """
    Prepares the grids and labels for training the model. If the user chooses not to use and wants to generate new ones
    from raw data, it first generates new grids and then prepares the dataloader.

Args:
    - batch_size (int): Size of the batches for training.
    - generate_grids (bool): Whether to generate new grids from raw data or load saved grids.
    - data_dir (str): Directory where raw LiDAR data or saved grids are stored. Default is data/raw.
    - grid_save_dir (str): Directory where generated grids are stored or will be saved. Default is data/pre_processed_data.
    - window_sizes (list): List of window sizes for grid generation (required if generating grids).
    - grid_resolution (int): Grid resolution (e.g., 128x128, required if generating grids).
    - channels (int): Number of channels in the grids (required if generating grids).
    - device (torch.device): Device to process data (CPU or GPU, required if generating grids).

    Returns:
    - DataLoader: A DataLoader object with the grids and their corresponding labels.
    """
    grids = []
    labels = []

    if generate_grids:
        # Process raw LiDAR data and generate grids
        print("Generating new grids from raw data...")
        data_array = read_las_file_to_numpy(data_dir)
        grids_dict = generate_multiscale_grids(data_array, window_sizes, grid_resolution, channels, grid_save_dir, device)

        # Loop through grids_dict to extract grids and labels
        for scale, data in grids_dict.items():
            grids.extend(data['grids'])
            labels.extend(data['class_labels'])
    else:
        # Load saved grids
        print("Loading saved grids...")
        for file_name in os.listdir(grid_save_dir):
            if file_name.endswith('.npy'):
                grid = np.load(os.path.join(grid_save_dir, file_name))
                grids.append(grid)

                # Extract label from filename (assuming format like grid_0_small_class_X.npy)
                label = int(file_name.split('_')[-1].split('.')[0].replace('class_', ''))
                labels.append(label)

    # Convert lists to tensors
    grids = torch.tensor(grids, dtype=torch.float32)  # Assuming float grids
    labels = torch.tensor(labels, dtype=torch.long)  # Assuming class labels are integers

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(grids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def save_model(model, save_dir):
    """
    Saves the MCNN model in the specified directory with a timestamp.

    Args:
    - model (nn.Module): The MCNN model to be saved.
    - save_dir (str): Directory where the model will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create a filename with the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(save_dir, f"mcnn_{timestamp}.pth")

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')



