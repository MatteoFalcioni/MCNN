import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import KDTree



def gpu_create_feature_grid(center_points, window_size, grid_resolution=128, channels=3, device=None):
    """
    Creates a batch of grids around the center points and initializes cells to store feature values.

    Args:
    - center_points (torch.Tensor): A tensor of shape [batch_size, 3] containing (x, y, z) coordinates of the center points.
    - window_size (float): The size of the square window around each center point (in meters).
    - grid_resolution (int): The number of cells in one dimension of the grid (e.g., 128 for a 128x128 grid).
    - channels (int): The number of channels in the resulting image. Default is 3 for RGB.
    - device (torch.device): The device (CPU or GPU) where tensors will be created.

    Returns:
    - grids (torch.Tensor): A tensor of shape [batch_size, channels, grid_resolution, grid_resolution].
    - cell_size (float): The size of each cell in meters.
    - x_coords (torch.Tensor): A tensor of shape [batch_size, grid_resolution] for x coordinates of grid cells.
    - y_coords (torch.Tensor): A tensor of shape [batch_size, grid_resolution] for y coordinates of grid cells.
    - constant_z (torch.Tensor): A tensor of shape [batch_size] containing the z coordinates of the center points.
    """

    center_points = center_points.to(device)    # additional check to avoid cpu usage
    batch_size = center_points.shape[0]  # Number of points in the batch

    # Calculate the size of each cell in meters
    cell_size = window_size / grid_resolution

    # Initialize the grids with zeros; one grid for each point in the batch
    grids = torch.zeros((batch_size, channels, grid_resolution, grid_resolution), device=device)

    half_resolution_plus_half = (grid_resolution / 2) + 0.5

    # Create x and y coordinate grids for each point in the batch
    x_coords = center_points[:, 0].unsqueeze(1) - (half_resolution_plus_half - torch.arange(grid_resolution, device=device).view(1, -1)) * cell_size
    y_coords = center_points[:, 1].unsqueeze(1) - (half_resolution_plus_half - torch.arange(grid_resolution, device=device).view(1, -1)) * cell_size
    constant_z = center_points[:, 2]  # This gives a tensor of shape [batch_size]
    

    return grids, cell_size, x_coords, y_coords, constant_z


def gpu_assign_features_to_grid(batch_data, grids, x_coords, y_coords, constant_z, full_data, tree, channels=3, device=None):
    """
    Assign features from the nearest point to each cell in the grid for a batch of points using KDTree.

    Args:
    - batch_data (torch.Tensor): A tensor of shape [batch_size, 2] representing the (x, y) coordinates of points in the batch.
    - grids (torch.Tensor): A tensor of shape [batch_size, channels, grid_resolution, grid_resolution] for (points, channels, rows, columns).
    - x_coords (torch.Tensor): A tensor of shape [batch_size, grid_resolution] containing x coordinates for each grid cell.
    - y_coords (torch.Tensor): A tensor of shape [batch_size, grid_resolution] containing y coordinates for each grid cell.
    - constant_z (torch.Tensor): A tensor of shape [batch_size] containing the z coordinates of the center points.
    - full_data (np.ndarray): The entire point cloud's data to build the KDTree for nearest neighbor search.
    - tree (KDTree): KDTree for efficient nearest-neighbor search.
    - channels (int): Number of feature channels in the grid (default is 3 for RGB).
    - device (torch.device): The device (CPU or GPU) to run this on.

    Returns:
    - grids (torch.Tensor): The updated grids with features assigned based on the nearest points from the full point cloud.
    """

    # Iterate through each batch point and its grid
    with torch.no_grad():

        batch_size = batch_data.shape[0]  # Number of points in the batch
        
        for i in range(batch_size):
            # Create a meshgrid for the current grid cell coordinates
            grid_x, grid_y = torch.meshgrid(x_coords[i], y_coords[i], indexing='ij')

            # Stack the x, y, and z coordinates together for the KDTree query
            grid_coords = torch.stack((grid_x.flatten(), grid_y.flatten(), constant_z[i].expand(grid_x.numel())), dim=-1).cpu().numpy()

            # Query the KDTree to find the nearest point in the full point cloud for each grid cell
            _, closest_points_idxs = tree.query(grid_coords)  

            # Assign features to the grid for the i-th batch based on the closest points from the full point cloud
            # Using a more efficient method to fill the grid
            for channel in range(channels):
                # Convert the selected features to a PyTorch tensor and move to the correct device
                features_to_assign = torch.tensor(full_data[closest_points_idxs, 3 + channel], dtype=torch.float32, device=device)
    
                # Assign all features for the current channel at once
                grids[i, channel].view(-1)[:] = features_to_assign  # Assign features for all cells at once


    return grids


def prepare_grids_dataloader(data_array, batch_size, num_workers):
    """
    Prepares the DataLoader for batching the point cloud data.

    Args:
    - data_array (np.ndarray): The dataset containing (x, y, z) coordinates, features, and class labels.
    - batch_size (int): The number of points to process in each batch.
    - num_workers (int): Number of workers for data loading.

    Returns:
    - data_loader (DataLoader): A DataLoader that batches the dataset.
    """
    
    # Create tensors for coordinates and labels
    coords_tensor = torch.tensor(data_array[:, :3], dtype=torch.float32)  # Shape: [num_points, 3]
    labels_tensor = torch.tensor(data_array[:, -1], dtype=torch.long)      # Shape: [num_points]
    
    # Combine them into a single tensor of shape [num_points, 4]
    combined_tensor = torch.empty((coords_tensor.size(0), 4), dtype=torch.float32)
    combined_tensor[:, :3] = coords_tensor  # First 3 columns are coordinates
    combined_tensor[:, 3] = labels_tensor.float()  # Last column is the label as float (for uniformity)

    # Create TensorDataset with the combined tensor
    dataset = TensorDataset(combined_tensor)

    # Create the DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return data_loader

    


def gpu_generate_multiscale_grids(data_loader, window_sizes, grid_resolution, channels, device, full_data, save_dir=None, save=False):
    """
    Generates grids for multiple scales (small, medium, large) for the entire dataset in batches.

    Args:
    - data_loader (DataLoader): A DataLoader that batches the unified dataset (coordinates + labels).
    - window_sizes (list of tuples): List of window sizes for each scale (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)]).
    - grid_resolution (int): The grid resolution (e.g., 128x128).
    - channels (int): Number of feature channels in the grid (e.g., 3 for RGB).
    - device (torch.device): The device to run on (CPU or GPU).
    - full_data (np.ndarray): The entire point cloud's data to build the KDTree for nearest neighbor search.
    - save_dir (str): Directory to save the generated grids (optional).
    - save (bool): Whether to save the grids to disk (default is False).

    Returns:
    - labeled_grids_dict (dict): Dictionary containing the generated grids and corresponding labels for each scale.
    """

    # Create a dictionary to hold grids and class labels for each scale
    labeled_grids_dict = {scale_label: {'grids': [], 'class_labels': []} for scale_label, _ in window_sizes}

    # Create the KDTree 
    print('creating KD tree...')
    tree = KDTree(full_data[:, :3])
    print('KD tree created succesfully.')


    # Iterate over the DataLoader batches
    for batch_idx, batch_data in enumerate(data_loader):   
        print(f"Processing batch {batch_idx + 1}/{len(data_loader)}...")

        # Access the batch data; it should be a single tensor with shape (batch_size, 4)
        batch_tensor = batch_data[0].to(device)  # Access the first (and only) element as the combined tensor
        
        # Extract coordinates and labels from the batch tensor
        coordinates = batch_tensor[:, :3]  # First 3 columns for (x, y, z)
        labels = batch_tensor[:, 3]  # Last column for labels
            

        # For each scale, generate the grids and assign features
        for size_label, window_size in window_sizes:
            print(f"Generating {size_label} grid for batch {batch_idx} with window size {window_size}...")

            # Create a batch of grids
            grids, _, x_coords, y_coords, constant_z = gpu_create_feature_grid(coordinates, window_size, grid_resolution, channels, device)

            # Assign features to the grids using the GPU-based KDTree
            grids = gpu_assign_features_to_grid(coordinates, grids, x_coords, y_coords, constant_z, full_data, tree, channels, device)

            # Append the grids and labels to the dictionary
            labeled_grids_dict[size_label]['grids'].append(grids.cpu().numpy())  # Store as numpy arrays
            labeled_grids_dict[size_label]['class_labels'].append(labels.cpu().numpy())

            # Save the grid if save_dir is provided
            if save and save_dir is not None:
                save_grid(grids, labels, batch_idx, size_label, save_dir)

        # Clear variables to free memory
        del batch_data, labels, grids, x_coords, y_coords

    return labeled_grids_dict


def save_grid(grids, batch_labels, batch_idx, size_label, save_dir):
    """
    Helper function to save grids to disk.

    Args:
    - grids (torch.Tensor): The grids to be saved.
    - batch_labels (torch.Tensor): The labels corresponding to each grid.
    - batch_idx (int): The current batch index.
    - size_label (str): Label for the grid scale ('small', 'medium', 'large').
    - save_dir (str): Directory to save the generated grids.
    """
    for i, (grid, label) in enumerate(zip(grids, batch_labels)):
        try:
            # Convert grid to numpy and save
            grid_with_features = grid.cpu().numpy()
            scale_dir = os.path.join(save_dir, size_label)
            os.makedirs(scale_dir, exist_ok=True)
            grid_filename = os.path.join(scale_dir, f"grid_{batch_idx}_{i}_{size_label}_class_{int(label)}.npy")
            np.save(grid_filename, grid_with_features)
        except Exception as e:
            print(f"Error saving grid {i} in batch {batch_idx}: {str(e)}")





