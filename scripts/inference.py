import torch
from scripts.point_cloud_to_image import generate_multiscale_grids, compute_point_cloud_bounds
from utils.point_cloud_data_utils import remap_labels
import numpy as np
import csv
from datetime import datetime
import os
from scipy.spatial import cKDTree


def inference(model, data_array, window_sizes, grid_resolution, feature_indices, device, save_file=None, subsample_size=200):
    """
    Perform inference with the MCNN model, generating grids from point cloud points and comparing predicted labels with known true labels.

    Args:
    - model (nn.Module): The trained MCNN model.
    - data_array (np.ndarray): Array of points from the point cloud on which we want to perform inference.
    - window_sizes (list of tuples): List of window sizes for each scale (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)]).
    - grid_resolution (int): Resolution of the grid (e.g., 128x128).
    - feature_indices (list): List of feature indices to be selected from the full list of features.
    - device (torch.device): The device (CPU or GPU) to perform inference on.
    - save_file (str): Path to the file where labels will be saved.
    - subsample_size (int): Number of points to randomly sample for inference.

    Returns:
    - predicted_labels (torch.Tensor): Predicted class labels.
    """
    
    # remap labels to match the remapping in the original data (it was eneded to use cross entropy loss)
    data_array, _ = remap_labels(data_array)

    # Build the KDTree once for the entire dataset
    kdtree = cKDTree(data_array[:, :3])

    # Compute point cloud bounds once
    point_cloud_bounds = compute_point_cloud_bounds(data_array)

    # Subsample points for inference
    subsample_indices = np.random.choice(len(data_array), subsample_size, replace=False)

    # Initialize lists to store predicted and true labels
    predicted_labels_list = []
    true_labels_list = []

    # Perform inference point by point
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients during inference
        for idx in subsample_indices:
            center_point = data_array[idx, :3]
            true_label = data_array[idx, -1]  # Get true label from the data array

            # Generate multiscale grids for this point
            grids_dict, skipped = generate_multiscale_grids(
                center_point=center_point,
                data_array=data_array,
                window_sizes=window_sizes,
                grid_resolution=grid_resolution,
                feature_indices=feature_indices,
                kdtree=kdtree,
                point_cloud_bounds=point_cloud_bounds
            )

            if skipped:
                continue  # Skip points with invalid grids

            # Convert grids to tensors and move to device
            small_grid = torch.tensor(grids_dict['small'], dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
            medium_grid = torch.tensor(grids_dict['medium'], dtype=torch.float32).unsqueeze(0).to(device)
            large_grid = torch.tensor(grids_dict['large'], dtype=torch.float32).unsqueeze(0).to(device)

            # Forward pass through the model
            outputs = model(small_grid, medium_grid, large_grid)

            # Get predicted class label (highest probability for each sample)
            _, predicted_label = torch.max(outputs, dim=1)

            # Append predicted and true labels to the lists
            predicted_labels_list.append(predicted_label.item())
            true_labels_list.append(true_label)

    # Optionally save the true and predicted labels to a file
    if save_file:
        os.makedirs(save_file, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Add timestamp to filename
        file_name, file_extension = os.path.splitext(save_file)
        save_file_with_timestamp = f"{file_name}_{timestamp}{file_extension}"

        with open(save_file_with_timestamp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['True Label', 'Predicted Label'])  # Header
            for true, pred in zip(true_labels_list, predicted_labels_list):
                writer.writerow([int(true), int(pred)])

    # Convert the predicted labels to a tensor and return
    predicted_labels_tensor = torch.tensor(predicted_labels_list, dtype=torch.long)

    return predicted_labels_tensor


