import torch
from scripts.point_cloud_to_image import generate_multiscale_grids
from utils.point_cloud_data_utils import load_features_used
import numpy as np
import csv
from datetime import datetime
import os


def inference(model, data_array, window_sizes, grid_resolution, device, true_labels=None, save_file=None, label_file=None, grid_save_dir="data/inference", subsample_size=200, features_file=None):
    """
    Perform inference with the MCNN model, generating grids from point cloud points,
    saving grids, and saving true and predicted labels to a file for later analysis.

    Args:
    - model (nn.Module): The trained MCNN model.
    - data_array (np.ndarray): Array of points from the point cloud on which we want to perform inference.
    - window_sizes (list of tuples): List of window sizes for each scale (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)]).
    - grid_resolution (int): Resolution of the grid (e.g., 128x128).
    - device (torch.device): The device (CPU or GPU) to perform inference on.
    - true_labels (torch.Tensor or np.ndarray): True class labels. 
    - save_file (str): Path to the file where labels will be saved.
    - label_file (str): Path to the CSV file containing the true labels.
    - grid_save_dir (str): Directory to save generated grids . Defaults to 'data/inference'.
    - subsample_size (int): Number of points to randomly sample for inference.
    - features_file (str): Path to the file containing the features used for grid generation.
    
    Returns:
    - predicted_labels (torch.Tensor): Predicted class labels.
    """

    # Ensure the grid save directory exists
    os.makedirs(grid_save_dir, exist_ok=True)

    # Load features used for grid generation
    if features_file:
        try:
            features_used = load_features_used(features_file)
            print(f"Loaded features used for grids: {features_used}")
        except Exception as e:
            raise ValueError(f"Failed to load features used from {features_file}: {e}")
    else:
        raise ValueError("Features file is required to ensure grid consistency.")

    # Generate multiscale grids for all points and save them
    generate_multiscale_grids(
        data_array=data_array,
        window_sizes=window_sizes,
        grid_resolution=grid_resolution,
        known_features=features_used,  # Use the same known features
        features_to_use=features_used,  # Ensure the same features are used for generation
        save_dir=grid_save_dir  # Save the grids to the inference directory
    )

    # If the true labels are provided via CSV file, load them
    if label_file:
        true_labels = []
        with open(label_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header if exists
            for row in reader:
                true_labels.append(int(row[0]))
        true_labels = np.array(true_labels)

    # Subsample points for inference
    subsample_indices = np.random.choice(len(data_array), subsample_size, replace=False)
    subsampled_true_labels = true_labels[subsample_indices] if true_labels is not None else None

    # Load the saved grids for the subsampled points
    grids_dict = load_saved_grids_for_subsample(grid_save_dir, subsample_indices)

    # Load the actual grid data from the file paths
    small_grids = torch.tensor(np.array([np.load(path) for path in grids_dict['small']['grids']]), dtype=torch.float32).to(device)
    medium_grids = torch.tensor(np.array([np.load(path) for path in grids_dict['medium']['grids']]), dtype=torch.float32).to(device)
    large_grids = torch.tensor(np.array([np.load(path) for path in grids_dict['large']['grids']]), dtype=torch.float32).to(device)

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():  # No need to track gradients during inference
        # Forward pass through the model
        outputs = model(small_grids, medium_grids, large_grids)

        # Get predicted class labels (highest probability for each sample)
        _, predicted_labels = torch.max(outputs, dim=1)

    # Save true and predicted labels to a file (if provided)
    if save_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Add timestamp to filename
        file_name, file_extension = os.path.splitext(save_file)
        save_file_with_timestamp = f"{file_name}_{timestamp}{file_extension}"

        with open(save_file_with_timestamp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['True Label', 'Predicted Label'])  # Header
            if subsampled_true_labels is not None:
                for true, pred in zip(subsampled_true_labels, predicted_labels.cpu().numpy()):
                    writer.writerow([int(true), int(pred)])
            else:
                for pred in predicted_labels.cpu().numpy():
                    writer.writerow(['N/A', int(pred)])  # In case true labels are not provided

    return predicted_labels


def load_saved_grids_for_subsample(grid_save_dir, subsample_indices):
    """
    Load saved grid file paths based on subsampled indices across 'small', 'medium', and 'large' scales.

    Args:
    - grid_save_dir (str): Directory where the grids are saved.
    - subsample_indices (list): List of indices to subsample.

    Returns:
    - grids_dict (dict): Dictionary containing the file paths for each grid.
    """
    grids_dict = {'small': {'grids': []}, 'medium': {'grids': []}, 'large': {'grids': []}}

    for idx in subsample_indices:
        small_path = os.path.join(grid_save_dir, 'small', f"grid_{idx}_small.npy")
        medium_path = os.path.join(grid_save_dir, 'medium', f"grid_{idx}_medium.npy")
        large_path = os.path.join(grid_save_dir, 'large', f"grid_{idx}_large.npy")

        grids_dict['small']['grids'].append(np.load(small_path))
        grids_dict['medium']['grids'].append(np.load(medium_path))
        grids_dict['large']['grids'].append(np.load(large_path))

    return grids_dict
