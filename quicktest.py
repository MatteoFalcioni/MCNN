import os
import numpy as np
from scripts.point_cloud_to_image import generate_multiscale_grids
from utils.point_cloud_data_utils import read_las_file_to_numpy, numpy_to_dataframe, remap_labels

# Directory to save the generated grids
save_dir = 'tests/test_grids'
os.makedirs(save_dir, exist_ok=True)

sample_size = 500
las_file_path = 'data/raw/labeled_FSL.las'
full_data, feature_names = read_las_file_to_numpy(las_file_path)
df = numpy_to_dataframe(data_array=full_data, feature_names=feature_names)

np.random.seed(42)  # For reproducibility
sampled_data = full_data[np.random.choice(full_data.shape[0], sample_size, replace=False)]
sampled_data, _ = remap_labels(sampled_data)

# Define the window sizes for multiscale grids
window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]

# Parameters for grid generation
grid_resolution = 128
channels = 10  # 10 feature channels

# Generate multiscale grids and save them to the 'tests/test_grids' folder
labeled_grids_dict = generate_multiscale_grids(
    data_array=sampled_data,
    window_sizes=window_sizes,
    grid_resolution=grid_resolution,
    channels=channels,
    save_dir=save_dir,
    save=True  # Save the grids
)

# Confirm the grids were generated and saved
for scale_label, content in labeled_grids_dict.items():
    print(f"Grids for {scale_label} scale saved in {os.path.join(save_dir, scale_label)}")
    print(f"Number of grids: {content['grids'].shape[0]}")
    print(f"Grid shape (channels x height x width): {content['grids'][0].shape}")

# Load one of the saved grids to confirm the shape
for scale_label in ['small', 'medium', 'large']:
    grid_file = os.path.join(save_dir, scale_label, 'grid_0_' + scale_label + '_class_0.npy')
    if os.path.exists(grid_file):
        loaded_grid = np.load(grid_file)
        print(f"Loaded {scale_label} grid from {grid_file}")
        print(f"Loaded grid shape: {loaded_grid.shape}")  # Check the shape after loading

