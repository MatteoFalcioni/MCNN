import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import numpy as np
from utils.point_cloud_data_utils import read_las_file_to_numpy, numpy_to_dataframe
from scripts.point_cloud_to_image import generate_multiscale_grids
from datetime import datetime
import pandas as pd


def prepare_dataloader(batch_size, pre_process_data, data_dir='data/raw/labeled_FSL.las', grid_save_dir='data/pre_processed_data',
                       window_sizes=None, grid_resolution=128, features_to_use=None, save_grids=True, train_split=0.8):
    """
        Prepares DataLoader objects for training and evaluation with three grid sizes (small, medium, large) and labels.

        If `pre_process_data` is True, the function will generate new grids from raw data. If False, it will load
        pre-saved grids from the specified directory.

        Args:
        - batch_size (int): Size of the batches for training and evaluation.
        - pre_process_data (bool): Whether to generate new grids from raw data or load pre-saved grids.
        - data_dir (str): File path to where raw (labeled) LiDAR data is stored if generating grids. Default is 'data/raw/features_F.las'.
        - grid_save_dir (str): Directory where the generated grids will be stored or loaded from. Default is 'data/pre_processed_data'.
        - window_sizes (list of tuples): List of window sizes for each scale (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)]).
        - grid_resolution (int): Resolution of the grid (e.g., 128x128). Required if generating grids.
        - features_to_use (list): List of feature names to use for each grid.
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
        if not window_sizes:
            raise ValueError("Window sizes must be provided when generating grids.")

        # Check if the data is already in .npy format
        if data_dir.endswith('.npy'):
            raise ValueError("Simple numpy files cannot be used for data preprocessing, feature names info is needed.")

        elif data_dir.endswith('.las'):
            print("Generating new grids from raw LAS data...")
            # For LAS files, the function returns both the data array and the known features
            data_array, known_features = read_las_file_to_numpy(data_dir, features_to_extract=features_to_use)
        
        elif data_dir.endswith('.csv'):
            print("Generaing new grids from raw CSV data...")

            # Read CSV into a DataFrame to extract column names
            df = pd.read_csv(data_dir)
            data_array = df.values
            known_features = df.columns.tolist()  # Extract column names as features

        # label remapping is in the following function
        grids_dict = generate_multiscale_grids(data_array, window_sizes, grid_resolution, features_to_use, known_features, grid_save_dir,
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
                    small_grids.append(torch.tensor(grid, dtype=torch.float64))
                elif scale == 'medium':
                    medium_grids.append(torch.tensor(grid, dtype=torch.float64))
                elif scale == 'large':
                    large_grids.append(torch.tensor(grid, dtype=torch.float64))

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

    if train_split > 0.0: 
        # Split the dataset into training and evaluation sets
        train_size = int(train_split * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

        # Create DataLoaders for training and evaluation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    else: 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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




