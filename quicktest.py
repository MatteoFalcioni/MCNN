from utils.point_cloud_data_utils import read_las_file_to_numpy, numpy_to_dataframe
from utils.plot_utils import visualize_grid, visualize_grid_with_comparison
import os

from data.transforms.point_cloud_to_image import create_feature_grid, assign_features_to_grid

file_path = os.path.abspath("data/raw/features_F.las")
numpy_data, feature_names = read_las_file_to_numpy(file_path)

# np.random.seed(42)  # For reproducibility
# subsample = numpy_data[np.random.choice(numpy_data.shape[0], 7064588, replace=False)]

window_size = 20.0
center_point = numpy_data[100000, :3]
grid, cell_size, x_coords, y_coords, z_coords = create_feature_grid(
            center_point, window_size=window_size, grid_resolution=128, channels=10
        )
print("grid created around center point")
assign_features_to_grid(numpy_data, grid, x_coords, y_coords, channels=10)
print("features assigned, feature image created")

df = numpy_to_dataframe(numpy_data, feature_names)
print("df has been created")

visualize_grid(grid, channel=8)
visualize_grid_with_comparison(grid, df, center_point, window_size=window_size, feature_names=feature_names, channel=8, visual_size=50)
