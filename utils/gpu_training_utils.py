from scripts.gpu_grid_gen import build_cuml_knn, generate_multiscale_grids_gpu
from torch.utils.data import Dataset, DataLoader, random_split
from scripts.point_cloud_to_image import compute_point_cloud_bounds
import torch
from utils.point_cloud_data_utils import read_file_to_numpy, remap_labels


class GPU_PointCloudDataset(Dataset):
    def __init__(self, data_array, window_sizes, grid_resolution, features_to_use, known_features):
        """
        Dataset class for streaming multiscale grid generation from point cloud data on the GPU.

        Args:
        - data_array (numpy.ndarray): The entire point cloud data array (already remapped).
        - window_sizes (list): List of tuples for grid window sizes (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)]).
        - grid_resolution (int): Grid resolution (e.g., 128x128).
        - features_to_use (list): List of feature names for generating grids.
        - known_features (list): All known feature names in the data array.
        """
        self.data_array = data_array
        self.window_sizes = window_sizes
        self.grid_resolution = grid_resolution
        self.features_to_use = features_to_use
        self.known_features = known_features
        
        # Build cuML KNN model on the GPU (use only the XYZ coordinates for KNN)
        self.gpu_tree = build_cuml_knn(data_array[:, :3])
        self.feature_indices = [known_features.index(feature) for feature in features_to_use]
        self.point_cloud_bounds = compute_point_cloud_bounds(data_array)

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        """
        Generates multiscale grids for the point at index `idx` and returns them as PyTorch tensors, along with the index.
        """
        # Extract the single point's data using `idx`
        center_point = self.data_array[idx, :3]  # Get the x, y, z coordinates
        label = self.data_array[idx, -1]  # Get the label for this point

        # Generate multiscale grids for this point using the GPU
        grids_dict, skipped = generate_multiscale_grids_gpu(
            center_point, data_array=self.data_array, window_sizes=self.window_sizes,
            grid_resolution=self.grid_resolution, feature_indices=self.feature_indices,
            cuml_knn=self.gpu_tree, point_cloud_bounds=self.point_cloud_bounds
        )

        if skipped:
            return None  # Skip this point if grid generation fails or if out of bounds

        # Convert grids to PyTorch tensors (channels first: C, H, W)
        small_grid = torch.tensor(grids_dict['small'], dtype=torch.float32)
        medium_grid = torch.tensor(grids_dict['medium'], dtype=torch.float32)
        large_grid = torch.tensor(grids_dict['large'], dtype=torch.float32)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        # Return the grids and label
        return small_grid, medium_grid, large_grid, label, idx
    

def gpu_custom_collate_fn(batch):
    """
    Custom collate function to filter out None values (skipped points) and ensure all data is on the GPU.
    """
    # Filter out any None values (i.e., skipped points)
    batch = [item for item in batch if item is not None]
    
    # If the batch is empty (all points were skipped), return None
    if len(batch) == 0:
        return None
    
    # Unpack the batch into grids and labels
    small_grids, medium_grids, large_grids, labels, indices = zip(*batch)
    
    # Stack the grids and labels to create tensors for the batch
    # Ensure the grids are on the correct device (GPU)
    device = small_grids[0].device  # Assuming all grids are on the same device
    small_grids = torch.stack(small_grids).to(device)
    medium_grids = torch.stack(medium_grids).to(device)
    large_grids = torch.stack(large_grids).to(device)
    labels = torch.stack(labels).to(device)
    indices = torch.tensor(indices).to(device)
    
    return small_grids, medium_grids, large_grids, labels, indices


def gpu_prepare_dataloader(batch_size, data_dir=None, 
                       window_sizes=None, grid_resolution=128, features_to_use=None, 
                       train_split=None, features_file_path=None, num_workers=4, shuffle_train=True, device='cuda'):
    """
    Prepares the DataLoader by loading the raw data and streaming multiscale grid generation on the GPU.
    
    Args:
    - batch_size (int): The batch size to be used for training.
    - data_dir (str): Path to the raw data (e.g., .las or .csv file). Default is None.
    - window_sizes (list): List of window sizes to use for grid generation. Default is None.
    - grid_resolution (int): Resolution of the grid (e.g., 128x128).
    - features_to_use (list): List of feature names to use for grid generation. Default is None.
    - train_split (float): Ratio of the data to use for training (e.g., 0.8 for 80% training data). Default is None.
    - features_file_path: File path to feature metadata, needed if using raw data in .npy format. Default is None.
    - num_workers (int): number of workers for parallelized process. Default is 4.
    - shuffle_train (bool): Whether to shuffle the data for training. Default is True.
    - device (str): The device ('cuda' or 'cpu') for tensor operations.

    Returns:
    - train_loader (DataLoader): DataLoader for training.
    - eval_loader (DataLoader): DataLoader for validation (if train_split is not None, else eval_loader=None).
    """
    
    # Check if data directory was passed as input
    if data_dir is None:
        raise ValueError('ERROR: Data directory was not passed as input to the dataloader.')

    # Read the raw point cloud data 
    data_array, known_features = read_file_to_numpy(data_dir=data_dir, features_to_use=None, features_file_path=features_file_path)

    # Remap labels to ensure they vary continuously (needed for CrossEntropyLoss)
    data_array, _ = remap_labels(data_array)

    # Create the dataset 
    full_dataset = GPU_PointCloudDataset(
        data_array=data_array,
        window_sizes=window_sizes,
        grid_resolution=grid_resolution,
        features_to_use=features_to_use,
        known_features=known_features,
    )

    # Split the dataset into training and evaluation sets (if train_split is provided)
    if train_split is not None:
        train_size = int(train_split * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

        # Create DataLoaders for training and evaluation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, collate_fn=gpu_custom_collate_fn, num_workers=num_workers)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpu_custom_collate_fn, num_workers=num_workers)
    else:
        # If no train/test split, create one DataLoader for the full dataset
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle_train, collate_fn=gpu_custom_collate_fn, num_workers=num_workers)
        eval_loader = None

    return train_loader, eval_loader

