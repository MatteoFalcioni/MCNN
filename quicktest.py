import os
import numpy as np
from scripts.point_cloud_to_image import generate_multiscale_grids
from utils.point_cloud_data_utils import read_las_file_to_numpy, numpy_to_dataframe, remap_labels

# Directory to save the generated grids
save_dir = 'tests/test_grids'
os.makedirs(save_dir, exist_ok=True)

# Load LAS file, get the data and feature names
sample_size = 500
las_file_path = 'data/raw/labeled_FSL.las'
full_data, feature_names = read_las_file_to_numpy(las_file_path)
df = numpy_to_dataframe(data_array=full_data, feature_names=feature_names)

np.random.seed(42)  # For reproducibility
sampled_data = full_data[np.random.choice(full_data.shape[0], sample_size, replace=False)]
sampled_data, _ = remap_labels(sampled_data)

# Define the window sizes for multiscale grids
window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
# Add dummy class labels (e.g., 0 or 1) to unlabeled data

# Parameters for grid generation
grid_resolution = 128
channels = 3  # Assuming 3 feature channels

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
