import torch
from scripts.point_cloud_to_image import generate_multiscale_grids
import numpy as np
import csv
from datetime import datetime


def inference(model, data_array, window_sizes, grid_resolution, channels, device, true_labels=None, save_file=None):
    """
    Perform inference with the MCNN model, generating grids from point cloud points,
    and save true and predicted labels to a file for later analysis.

    Args:
    - model (nn.Module): The trained MCNN model.
    - data_array (np.ndarray): Array of points from the point cloud on which we want to perform inference.
    - window_sizes (list of tuples): List of window sizes for each scale (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)]).
    - grid_resolution (int): Resolution of the grid (e.g., 128x128).
    - channels (int): Number of channels in the grid.
    - device (torch.device): The device (CPU or GPU) to perform inference on.
    - true_labels (torch.Tensor or np.ndarray): True class labels (optional).
    - save_file (str): Path to the file where labels will be saved.

    Returns:
    - predicted_labels (torch.Tensor): Predicted class labels.
    """
    # Generate multiscale grids for the point cloud
    grids_dict = generate_multiscale_grids(
        data_array=data_array,
        window_sizes=window_sizes,
        grid_resolution=grid_resolution,
        channels=channels,
        save=False  # No need to save grids during inference
    )

    # Extract small, medium, and large grids from the dictionary
    small_grids = torch.tensor(np.array(grids_dict['small']['grids']), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    medium_grids = torch.tensor(np.array(grids_dict['medium']['grids']), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    large_grids = torch.tensor(np.array(grids_dict['large']['grids']), dtype=torch.float32).permute(0, 3, 1, 2).to(device)

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():  # No need to track gradients during inference
        # Forward pass through the model
        outputs = model(small_grids, medium_grids, large_grids)

        # Get predicted class labels (highest probability for each sample)
        _, predicted_labels = torch.max(outputs, dim=1)

    # Save true and predicted labels to a file (if provided)
    if save_file:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    # timestamp of year, month, day _ hour, minute, second
        file_name, file_extension = os.path.splitext(save_file)
        save_file_with_timestamp = f"{file_name}_{timestamp}{file_extension}"

        with open(save_file_with_timestamp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['True Label', 'Predicted Label'])  # Header
            if true_labels is not None:
                for true, pred in zip(true_labels, predicted_labels.cpu().numpy()):
                    writer.writerow([int(true), int(pred)])
            else:
                for pred in predicted_labels.cpu().numpy():
                    writer.writerow(['N/A', int(pred)])  # In case true labels are not provided

    return predicted_labels

