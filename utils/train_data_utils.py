import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels, clean_nan_values, apply_masks_KDTree
from scripts.point_cloud_to_image import generate_multiscale_grids_masked
from datetime import datetime
import pandas as pd
from scipy.spatial import cKDTree
import csv
from models.mcnn import MultiScaleCNN
import ast
import numpy as np


class PointCloudDataset(Dataset):
    def __init__(self, full_data_array, window_sizes, grid_resolution, features_to_use, known_features, subset_file=None):
        """
        Dataset class for streaming multiscale grid generation from point cloud data.

        Args:
        - full_data_array (numpy.ndarray): The entire point cloud data array (already remapped).
        - window_sizes (list): List of tuples for grid window sizes (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)]).
        - grid_resolution (int): Grid resolution (e.g., 128x128).
        - features_to_use (list): List of feature names for generating grids.
        - known_features (list): All known feature names in the data array.
        - subset_file (str, optional): Path to a csv file containing coordinates of points to be selected from the full data. If None, all are selected.
        """
        self.full_data_array = full_data_array
        self.window_sizes = window_sizes
        self.grid_resolution = grid_resolution
        self.features_to_use = features_to_use
        self.known_features = known_features
        
        # Build KDTree once for the entire dataset
        self.kdtree = cKDTree(full_data_array[:, :3])  # Use full data coordinates for KDTree (use full point cloud for neighbors feature assignment)
        self.feature_indices = [known_features.index(feature) for feature in features_to_use]
        
        # Apply masking and compute bounds
        self.selected_array, mask, point_cloud_bounds = apply_masks_KDTree(
            full_data_array=full_data_array,
            window_sizes=window_sizes,
            subset_file=subset_file
        )

        self.original_indices = np.where(mask)[0]
    

    def __len__(self):
        return len(self.selected_array)

    def __getitem__(self, idx):
        """
        Generates multiscale grids for the point at index `idx` and returns them as PyTorch tensors, along with the index.
        """
        # Extract the single point's data from the selected array using `idx`
        center_point = self.selected_array[idx, :3]  # Get the x, y, z coordinates
        label = self.selected_array[idx, -1]  # Get the label for this point

        # Generate multiscale grids for this point using the entire dataset for feature assignment
        grids_dict = generate_multiscale_grids_masked(center_point, 
                                                      data_array=self.full_data_array, 
                                                      window_sizes=self.window_sizes, 
                                                      grid_resolution=self.grid_resolution, 
                                                      feature_indices=self.feature_indices, 
                                                      kdtree=self.kdtree) 
        
        # Convert grids to PyTorch tensors
        small_grid = torch.tensor(grids_dict['small'], dtype=torch.float32)
        medium_grid = torch.tensor(grids_dict['medium'], dtype=torch.float32)
        large_grid = torch.tensor(grids_dict['large'], dtype=torch.float32)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
        
        # Return the grids, label, and original index 
        original_idx = self.original_indices[idx]  # Map back to the original data_array index

        # Return the grids and label
        return small_grid, medium_grid, large_grid, label, original_idx
    

def prepare_dataloader(batch_size, data_filepath=None, 
                       window_sizes=None, grid_resolution=128, features_to_use=None, 
                       train_split=None, features_file_path=None, num_workers=4, shuffle_train=True, subset_file=None):
    """
    Prepares the DataLoader by loading the raw data and streaming multiscale grid generation.
    
    Args:
    - batch_size (int): The batch size to be used for training.
    - data_filepath (str): Path to the raw data (e.g., .las or .csv file). Default is None.
    - window_sizes (list): List of window sizes to use for grid generation. Default is None.
    - grid_resolution (int): Resolution of the grid (e.g., 128x128).
    - features_to_use (list): List of feature names to use for grid generation. Default is None.
    - train_split (float): Ratio of the data to use for training (e.g., 0.8 for 80% training data). Default is None.
    - features_file_path: File path to feature metadata, needed if using raw data in .npy format. Default is None.
    - num_workers (int): number of workers for parallelized process. Default is 4.
    - shuffle_train (bool): Whether to shuffle the data for training. Default is True.
    - subset_file (str, optional): Path to a csv file containing coordinates of points to be selected from the full data. If None, all are selected.

    Returns:
    - train_loader (DataLoader): DataLoader for training.
    - eval_loader (DataLoader): DataLoader for validation (if train_split is not None, else eval_loader=None).
    """
    
    # Check if data directory was passed as input
    if data_filepath is None:
        raise ValueError('ERROR: Data filepath was not passed as input to the dataloader.')

    # Read the raw point cloud data 
    data_array, known_features = read_file_to_numpy(data_dir=data_filepath, features_to_use=None, features_file_path=features_file_path)

    # Remap labels to ensure they vary continuously (needed for CrossEntropyLoss)
    data_array, _ = remap_labels(data_array=data_array)
    # clean data fom nan/inf values (replace them w/ 0.0)
    data_array = clean_nan_values(data_array=data_array)

    # Create the dataset 
    full_dataset = PointCloudDataset(
        full_data_array=data_array,
        window_sizes=window_sizes,
        grid_resolution=grid_resolution,
        features_to_use=features_to_use,
        known_features=known_features,
        subset_file=subset_file
    )

    # Split the dataset into training and evaluation sets (if train_split is provided)
    if train_split is not None:
        train_size = int(train_split * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

        # Create DataLoaders for training and evaluation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers) #collate_fn=custom_collate_fn,
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        # If no train/test split, create one DataLoader for the full dataset
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        eval_loader = None

    return train_loader, eval_loader


def save_model(model, save_dir='models/saved', used_features=None, hyperparameters=None):
    """
    Saves the PyTorch model along with the features and hyperparameters used in training.

    Args:
    - model (nn.Module): The trained model to be saved.
    - save_dir (str): The directory where the model and parameters will be saved. Default is 'models/saved'.
    - used_features (list): List of features used for training the model.
    - hyperparameters (dict): Dictionary of hyperparameters used during training.

    Returns:
    - model_save_foder (str): Name of the folder where the model is saved. Needed later for inference.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get the current date and time
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create the model folder with the timestamp
    model_save_folder = os.path.join(save_dir, f"mcnn_model_{current_time}")
    os.makedirs(model_save_folder, exist_ok=True)

    # Save the model in the folder
    model_filename = f"model.pth"
    model_save_path = os.path.join(model_save_folder, model_filename)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # Save the used features and hyperparameters
    if used_features or hyperparameters:
        save_used_parameters(used_features, hyperparameters, model_save_folder)

    return model_save_folder


def load_model(model_path, device, num_channels):
    """
    Loads a saved PyTorch model, initializes it, and sets it to evaluation mode.
    Retrieves num_classes from the hyperparameters.csv file stored in the model folder.
    
    Args:
    - model_path (str): Path to the saved model state dictionary.
    - device (torch.device): Device where the model will be loaded (CPU or GPU).
    - num_channels (int): Number of input channels for the model.
    
    Returns:
    - model (torch.nn.Module): The loaded and initialized model set to evaluation mode.
    """
    print(f'Loading pre-trained model from path: {model_path}\n')

    # Determine the path to the hyperparameters file (same directory as model)
    model_dir = os.path.dirname(model_path)
    hyperparameters_file = os.path.join(model_dir, 'hyperparameters.csv')

    # Read the hyperparameters file to get num_classes
    num_classes = None
    if os.path.exists(hyperparameters_file):
        with open(hyperparameters_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                param, value = row
                if param == 'num_classes':
                    num_classes = int(value)
                    break

    if num_classes is None:
        raise ValueError("num_classes not found in hyperparameters.csv")
    
    # Initialize the model
    model = MultiScaleCNN(channels=num_channels, classes=num_classes).to(device)
    
    # Load the saved model state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # Set the model to evaluation mode
    model.eval()

    print('Model loaded successfully\n')
    
    return model
    
    
def save_used_parameters(used_features=None, hyperparameters=None, save_dir='models/saved'):
    """
    Saves the features and hyperparameters used during training to a CSV file.

    Args:
    - used_features (list): List of features used in training.
    - hyperparameters (dict): Dictionary (name, value) of hyperparameters used during training. 
    - save_dir (str): Directory where the used features and the model's hyperparameters are to be saved.

    Returns:
    - None
    """

    # Save the features
    if used_features is not None:
        feature_file = os.path.join(save_dir, 'features_used.csv')
        with open(feature_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Used Features'])  # Header
            writer.writerow(used_features)
        print(f"Features saved to {feature_file}")
    else:
        print('used features were not specified in saving, so they could not be saved together with the model.')

    # Save the hyperparameters
    if hyperparameters is not None:
        hyperparameters_file = os.path.join(save_dir, 'hyperparameters.csv')
        with open(hyperparameters_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Hyperparameter', 'Value'])  # Header
            for param, value in hyperparameters.items():
                writer.writerow([param, value])
        print(f"Hyperparameters saved to {hyperparameters_file}")
    else:
        print('hyperparameters were not specified in saving, so they could not be saved together with the model.')


def load_parameters(model_folder):
    """
    Loads the features used for training and window sizes from saved CSV files.

    Args:
    - model_folder (str): Path to the folder containing the saved model and metadata.

    Returns:
    - features_list (list): List of features loaded from the CSV file.
    - num_loaded_channels (int): Number of loaded features.
    - window_sizes (list): Parsed list of window size tuples (e.g., [('small', 10.0), ('medium', 20.0), ('large', 30.0)]).
    """
    
    # Check that model_folder is the actual directory and not a filepath to the model
    if model_folder.endswith('.pth'):
        model_folder = os.path.dirname(model_folder)

    # Check if the folder exists
    if not os.path.exists(model_folder):
        raise FileNotFoundError(f"The specified model folder '{model_folder}' does not exist.")

    # Construct the paths to the relevant files
    features_file_path = os.path.join(model_folder, 'features_used.csv')
    hyperparameters_file_path = os.path.join(model_folder, 'hyperparameters.csv')

    # Check if the file exists
    if not os.path.exists(features_file_path):
        raise FileNotFoundError(f"Features file not found at {features_file_path}")

    # Load the features from the CSV file, skipping the header
    try:
        with open(features_file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            features_list = next(reader)  # Load the actual features row
            num_loaded_channels = len(features_list)

        if not features_list:
            raise ValueError(f"The features file at {features_file_path} is empty or invalid.")

    except Exception as e:
        raise ValueError(f"Error loading features from {features_file_path}: {e}")


    # Load window_sizes from hyperparameters.csv
    if not os.path.exists(hyperparameters_file_path):
        raise FileNotFoundError(f"Hyperparameters file not found at {hyperparameters_file_path}")

    try:
        window_sizes = None
        with open(hyperparameters_file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            
            for row in reader:
                if len(row) == 2:  # Ensure it's a key-value pair
                    key, value = row
                    if key == "window_sizes":
                        window_sizes = ast.literal_eval(value)  # Safely evaluate the string
                        
        if window_sizes is None:
            raise ValueError(f"'window_sizes' not found in {hyperparameters_file_path}.")
    except Exception as e:
        raise ValueError(f"Error loading window_sizes from {hyperparameters_file_path}: {e}")

    return features_list, num_loaded_channels, window_sizes


