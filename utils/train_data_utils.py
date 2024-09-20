import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from utils.point_cloud_data_utils import read_las_file_to_numpy
from scripts.point_cloud_to_image import generate_multiscale_grids


def prepare_dataloader(batch_size, pre_process_data, data_dir='data/raw', grid_save_dir='data/pre_processed_data',
                       window_sizes=None, grid_resolution=None, channels=None, device=None, save_grids=True):
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

