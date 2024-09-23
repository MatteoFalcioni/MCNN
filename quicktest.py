from utils.point_cloud_data_utils import read_las_file_to_numpy, numpy_to_dataframe, remap_labels
from scripts.point_cloud_to_image import generate_multiscale_grids
from utils.config_handler import parse_arguments
import numpy as np
import time
from scripts.inference import inference
from models.mcnn import MultiScaleCNN
import torch
from utils.device_utils import select_device


labeled_filepath = 'data/raw/labeled_FSL.las'
data_array, feature_names = read_las_file_to_numpy('data/raw/labeled_FSL.las')

# print(f'data array shape: {data_array.shape[1]}')

remapped_array, changes_dict = remap_labels(data_array)
sampled_array = remapped_array[np.random.choice(data_array.shape[0], 10000, replace=False)]

args = parse_arguments()

start_time = time.time()
generate_multiscale_grids(data_array=sampled_array, window_sizes=args.windows_sizes, grid_resolution=128, channels=10, save_dir='tests/test_grid_gen_onPC', save=True)
end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")

"""remapped_df = numpy_to_dataframe(remapped_array, feature_names)

labels = data_array[:, -1]  # If labels are in the last column


print(f'sample array:{sampled_array}')
sampled_df = numpy_to_dataframe(sampled_array, feature_names)

# Inspect the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(remapped_df.head())

# Inspect the column names and data types
print("\nColumn names and data types:")
print(remapped_df.dtypes)
# Count the unique classes
unique_classes = np.unique(labels)
num_classes = len(unique_classes)

print(changes_dict)

# Print the result
print(f'Number of unique classes: {num_classes}')
print(f'Classes: {unique_classes}')

print(f'sampled dataframe:')
print(sampled_df.head())"""

"""# Generate multiscale grids for the sampled data and save them
grids_dict = generate_multiscale_grids(data_array=sampled_array,
                                       window_sizes=[('small', 2.5), ('medium', 5.0), ('large', 10.0)],
                                       grid_resolution=128, channels=10,
                                       save_dir='data/pre_processed_data/', save=True)
model_path = 'models/saved/mcnn_model_20240922_231624.pth'

data_array, _ = read_las_file_to_numpy(labeled_filepath)
# need to remap labels to match the ones in training. Maybe consider remapping already when doing las -> numpy ?
remapped_array, _ = remap_labels(data_array)
sample_array = remapped_array[np.random.choice(data_array.shape[0], 20, replace=False)]
ground_truth_labels = sample_array[:, -1]  # Assuming labels are in the last column

device = select_device()
loaded_model = MultiScaleCNN(channels=10, classes=5).to(device)
loaded_model.load_state_dict(torch.load(model_path, map_location=device))

_, precision = inference(loaded_model, channels=10, grid_resolution=128, window_sizes=[('small', 2.5), ('medium', 5.0), ('large', 10.0)], data_array=sample_array, true_labels=ground_truth_labels, device=device)

print(precision)"""





