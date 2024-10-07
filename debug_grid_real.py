import numpy as np
import torch
import pandas as pd  # Import pandas for DataFrame
from scripts.optimized_pc_to_img import gpu_generate_multiscale_grids, prepare_grids_dataloader, gpu_assign_features_to_grid, gpu_create_feature_grid
from scripts.point_cloud_to_image import generate_multiscale_grids, assign_features_to_grid, create_feature_grid, compute_point_cloud_bounds
from utils.point_cloud_data_utils import read_las_file_to_numpy, remap_labels
from scipy.spatial import cKDTree as KDTree
from utils.plot_utils import visualize_grid_with_comparison, visualize_grid


# -------------------------------- GPU grid coordinates and stacking debug, trying to work in parallel ---------------------------------

def debug_create_feature_grid(center_point, window_size, grid_resolution=128, channels=3):
    """
    Debug version of create_feature_grid to print intermediate values.
    """
    # Calculate the size of each cell in meters
    cell_size = window_size / grid_resolution
    print(f"[DEBUG - CPU] Cell size: {cell_size}")
    
    # Initialize the grid to zeros
    grid = np.zeros((grid_resolution, grid_resolution, channels))

    # Generate cell coordinates for the grid based on the center point
    i_indices = np.arange(grid_resolution)
    j_indices = np.arange(grid_resolution)
    
    half_resolution_minus_half = (grid_resolution / 2) - 0.5

    x_coords = center_point[0] - (half_resolution_minus_half - j_indices) * cell_size
    y_coords = center_point[1] - (half_resolution_minus_half - i_indices) * cell_size
    
    print(f"[DEBUG - CPU] X-coordinates (sample): {x_coords[:5]} ... {x_coords[-5:]}")
    print(f"[DEBUG - CPU] Y-coordinates (sample): {y_coords[:5]} ... {y_coords[-5:]}")
    print(f"[DEBUG - CPU] Grid shape: {grid.shape}")

    constant_z = center_point[2]  # Z coordinate is constant for all cells

    return grid, cell_size, x_coords, y_coords, constant_z


def debug_gpu_create_feature_grid(center_points, window_size, grid_resolution=128, channels=3, device=None):
    """
    Creates a grid for features and 3D coordinates for each cell using meshgrid, allocating space for channels.

    Args:
    - center_points (torch.Tensor): A tensor of shape [batch_size, 3] containing (x, y, z) coordinates of the center points.
    - window_size (float): The size of the square window around each center point (in meters).
    - grid_resolution (int): The number of cells in one dimension of the grid (e.g., 128 for a 128x128 grid).
    - channels (int): The number of channels to store features in the grid.
    - device (torch.device): The device (CPU or GPU) where tensors will be created.

    Returns:
    - grids (torch.Tensor): A tensor of shape [batch_size, channels, grid_resolution, grid_resolution] initialized to zero.
    - cell_size (float): The size of each cell in meters.
    - grid_coords (torch.Tensor): A tensor of shape [batch_size, 3, grid_resolution, grid_resolution] containing 3D coordinates for each cell.
    """
    center_points = center_points.to(device)
    batch_size = center_points.shape[0]

    # Calculate the size of each cell in meters
    cell_size = torch.tensor(float(window_size) / float(grid_resolution), dtype=torch.float64, device=device)

    # Initialize the grids with zeros; one grid for each point in the batch
    grids = torch.zeros((batch_size, channels, grid_resolution, grid_resolution), dtype=torch.float64, device=device)

    # Calculate the coordinates of the grid cells
    half_resolution_minus_half = torch.tensor((grid_resolution / 2) - 0.5, dtype=torch.float64, device=device)


    # Use meshgrid to create the x and y coordinates for each grid
    x_range = torch.arange(grid_resolution, dtype=torch.float64, device=device)  # Shape: [grid_resolution]
    y_range = torch.arange(grid_resolution, dtype=torch.float64, device=device)  # Shape: [grid_resolution]

    # Create 2D grid coordinates using meshgrid
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')  # Shape: [grid_resolution, grid_resolution]

    # Compute the actual x and y coordinates for each batch element
    grid_coords_list = []
    for i in range(batch_size):
        x_coords = center_points[i, 0] - (half_resolution_minus_half - x_grid) * cell_size
        y_coords = center_points[i, 1] - (half_resolution_minus_half - y_grid) * cell_size
        z_coords = center_points[i, 2].expand(grid_resolution, grid_resolution)  # Constant z for all cells

        # Stack to get [3, grid_resolution, grid_resolution]
        grid_coords = torch.stack((x_coords, y_coords, z_coords), dim=0)
        grid_coords_list.append(grid_coords)

        # Debug: Print the range of coordinates for this batch
        print(f"[DEBUG - GPU] Batch {i} x_coords range: ({x_coords.min().item()}, {x_coords.max().item()})")
        print(f"[DEBUG - GPU] Batch {i} y_coords range: ({y_coords.min().item()}, {y_coords.max().item()})")
        print(f"[DEBUG - GPU] Batch {i} z_coords range: ({z_coords.min().item()}, {z_coords.max().item()})")

    # Combine into a single tensor of shape [batch_size, 3, grid_resolution, grid_resolution]
    grid_coords_combined = torch.stack(grid_coords_list)

    return grids, cell_size, grid_coords_combined

# CREATION OF FEATURE GRID IS ALRIGHT! ---------------------------------- 
# ------------------------------------------------DEBUGGING KD TREE------------------------------------------------

def debug_kd_tree_cpu(data_array, x_coords, y_coords, constant_z):
    """
    Debug version of KDTree query on the CPU using nested loops.

    Args:
    - data_array (numpy.ndarray): Point cloud data of shape (N, 3).
    - x_coords (numpy.ndarray): Array of x coordinates for grid cells.
    - y_coords (numpy.ndarray): Array of y coordinates for grid cells.
    - constant_z (float): The constant z coordinate for the grid cells.

    Returns:
    - indices (list): Indices of the nearest points in the point cloud for each grid cell.
    """
    tree = KDTree(data_array[:, :3].astype(np.float64))
    indices = []

    for i in range(len(x_coords)):
        for j in range(len(y_coords)):
            _, idx = tree.query([x_coords[i].astype(np.float64), y_coords[j].astype(np.float64), constant_z.astype(np.float64)])
            indices.append(idx)

    # Print the coordinates associated with the indices
    coords_from_indices = tree.data[indices]
    print(f"[DEBUG - CPU] KDTree indices (sample): {indices[:20]}")
    print(f"[DEBUG - CPU] Coordinates from indices (sample): {coords_from_indices[:20]}")

    return indices


def debug_kd_tree_cpu_bulk(data_array, x_coords, y_coords, constant_z):
    """
    Bulk version of KDTree query on the CPU.

    Args:
    - data_array (numpy.ndarray): Point cloud data of shape (N, 3).
    - x_coords (numpy.ndarray): Array of x coordinates for grid cells.
    - y_coords (numpy.ndarray): Array of y coordinates for grid cells.
    - constant_z (float): The constant z coordinate for the grid cells.

    Returns:
    - indices (numpy.ndarray): Indices of the nearest points in the point cloud for each grid cell.
    """
    # Create KDTree using point cloud data
    tree = KDTree(data_array[:, :3])

    # Generate grid coordinates for all cells in bulk
    grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')
    grid_coords = np.stack((grid_x.flatten(), grid_y.flatten(), np.full(grid_x.size, constant_z)), axis=-1)

    # Query the KDTree in bulk using all grid coordinates
    _, indices = tree.query(grid_coords)

    # Debug print for the indices
    print(f"[DEBUG - CPU Bulk] KDTree indices (sample): {indices[:20]}")
    
    # Print the coordinates associated with the indices
    coords_from_indices = tree.data[indices]
    print(f"[DEBUG - CPU Bulk] Coordinates from indices (sample): {coords_from_indices[:20]}")

    return indices


def debug_kd_tree_gpu(data_array, grid_coords, device):
    """
    Debug version of KDTree query on the GPU with enforced float64 precision.

    Args:
    - data_array (numpy.ndarray): Point cloud data of shape (N, 3).
    - grid_coords (torch.Tensor): Tensor of shape [batch_size, 3, grid_resolution, grid_resolution].

    Returns:
    - all_indices (list): List of indices of the nearest points for each grid cell in each batch.
    """
    # Create KDTree using point cloud data (enforcing float64)
    tree = KDTree(data_array[:, :3].astype(np.float64))

    batch_size = grid_coords.shape[0]
    all_indices = []

    # Iterate over each batch element
    for i in range(batch_size):
        # Extract the grid coordinates for the current batch element
        grid_coords_flat = grid_coords[i].permute(1, 2, 0).reshape(-1, 3).cpu().numpy().astype(np.float64)

        # Debug: Print grid coordinates range before KDTree query
        print(f"[DEBUG - GPU] Batch {i} grid_coords range: ({grid_coords_flat.min(axis=0)}, {grid_coords_flat.max(axis=0)})")
        
        # Query the KDTree using the current grid's coordinates
        _, indices = tree.query(grid_coords_flat)
        
        # Collect the indices for this batch
        all_indices.append(indices)

        # Debug print for the current batch's grid coordinates and indices
        print(f"[DEBUG - GPU] Grid coordinates (batch {i} sample): {grid_coords_flat[:10]}")
        coords_from_indices = tree.data[indices]
        print(f"[DEBUG - GPU] KDTree indices (batch {i} sample): {indices[:20]}")
        print(f"[DEBUG - GPU] Coordinates from indices (batch {i} sample): {coords_from_indices[:20]}")

        # Calculate and print the differences between the grid coordinates and matched points
        diff = grid_coords_flat[:20] - coords_from_indices[:20]
        print(f"[DEBUG - GPU] Differences (sample): {diff}")
        print(f"[DEBUG - GPU] Magnitude of differences (sample): {np.linalg.norm(diff, axis=1)}")


    
    return all_indices

# --------------------------------- STILL NOT SOLVED: SMALL DIFFERENCES BETWEEN COORDINATES (AND THUS, INDICES). maybe a Torch!=Numpy problem? ---------------------

# -----------------------------------------------------------DEBUGGING FEATURE ASSIGNMENT---------------------------------------------------------------------

def assign_features_to_grid(tree, data_array, grid, x_coords, y_coords, constant_z, feature_indices):
    """
    Assigns features from the nearest point in the dataset to each cell in the grid using a pre-built KDTree.

    Args:
    - tree (KDTree): Pre-built KDTree for efficient nearest-neighbor search.
    - data_array (numpy.ndarray): Array where each row represents a point with its x, y, z coordinates and features.
    - grid (numpy.ndarray): A 2D grid initialized to zeros, which will store feature values.
    - x_coords (numpy.ndarray): Array of x coordinates for the centers of the grid cells.
    - y_coords (numpy.ndarray): Array of y coordinates for the centers of the grid cells.
    - constant_z (float): The fixed z coordinate for the grid cells.
    - feature_indices (list): List of indices for the selected features.

    Returns:
    - grid (numpy.ndarray): The grid populated with the nearest point's feature values.
    """
    for i in range(len(x_coords)):
        for j in range(len(y_coords)):
            # Find the nearest point to the cell center (x_coords[i], y_coords[j], constant_z)
            _, idx = tree.query([x_coords[i], y_coords[j], constant_z])

            # Assign the features of the nearest point to the grid cell
            grid[i, j, :] = data_array[idx, feature_indices]

    return grid


"""def assign_features_to_grid_bulk(tree, data_array, grid, grid_coords_flat, feature_indices):

    # Perform bulk KDTree query
    _, indices = tree.query(grid_coords_flat)

    # Assign the features of the nearest points to the grid cells
    for k, idx in enumerate(indices):
        i, j = divmod(k, grid.shape[1])  # Convert flattened index to 2D grid index
        grid[i, j, :] = data_array[idx, feature_indices]

    return grid"""


def assign_features_to_grid_bulk(tree, data_array, grid, x_coords, y_coords, constant_z, feature_indices):
    """
    Assigns features to the grid using a bulk KDTree query (without nested loops).

    Args:
    - tree (KDTree): Pre-built KDTree for efficient nearest-neighbor search.
    - data_array (numpy.ndarray): Array where each row represents a point with its x, y, z coordinates and features.
    - grid (numpy.ndarray): A 2D grid initialized to zeros, which will store feature values.
    - x_coords (numpy.ndarray): Array of x coordinates for the centers of the grid cells.
    - y_coords (numpy.ndarray): Array of y coordinates for the centers of the grid cells.
    - constant_z (float): The fixed z coordinate for the grid cells.
    - feature_indices (list): List of indices for the selected features.

    Returns:
    - grid (numpy.ndarray): The grid populated with the nearest point's feature values.
    """
    # Generate grid coordinates for all cells in bulk
    grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')
    grid_coords = np.stack((grid_x.flatten(), grid_y.flatten(), np.full(grid_x.size, constant_z)), axis=-1)

    # Query the KDTree in bulk using all grid coordinates
    _, indices = tree.query(grid_coords)

    # Assign features using the bulk indices
    grid[:, :, :] = data_array[indices, :][:, feature_indices].reshape(grid.shape)

    return grid

def compare_feature_assignments(tree, data_array, x_coords, y_coords, constant_z, feature_indices, grid_resolution):
    """
    Compares the feature assignment between nested loop and bulk KDTree querying.

    Args:
    - tree (KDTree): Pre-built KDTree for efficient nearest-neighbor search.
    - data_array (numpy.ndarray): Array where each row represents a point with its x, y, z coordinates and features.
    - x_coords (numpy.ndarray): Array of x coordinates for the centers of the grid cells.
    - y_coords (numpy.ndarray): Array of y coordinates for the centers of the grid cells.
    - constant_z (float): The fixed z coordinate for the grid cells.
    - feature_indices (list): List of indices for the selected features.
    - grid_resolution (int): Resolution of the grid (number of cells in one dimension).

    Returns:
    - None
    """
    # Initialize grids
    grid_nested = np.zeros((grid_resolution, grid_resolution, len(feature_indices)), dtype=np.float64)
    grid_bulk = np.zeros_like(grid_nested)

    # Perform feature assignment using the nested loop
    grid_nested = assign_features_to_grid(tree, data_array, grid_nested, x_coords, y_coords, constant_z, feature_indices)

    # Perform feature assignment using the bulk method
    grid_bulk = assign_features_to_grid_bulk(tree, data_array, grid_bulk, x_coords, y_coords, constant_z, feature_indices)

    # Compare grids
    if np.allclose(grid_nested, grid_bulk, atol=1e-5):
        print("[SUCCESS] The grids match between nested loop and bulk KDTree querying.")
    else:
        print("[ERROR] The grids do not match between nested loop and bulk KDTree querying.")

    grid_nested = np.transpose(grid_nested, (2,0,1))
    grid_bulk = np.transpose(grid_bulk, (2,0,1))
    # Visualize the grids
    visualize_grid(grid_nested, channel=2, title="Grid with Nested Loop", feature_names=feature_names)
    visualize_grid(grid_bulk, channel=2, title="Grid with Bulk Querying", feature_names=feature_names)


def assign_features_to_grid_gpu(data_array, grid_coords, feature_indices, device):
    """
    Assigns features to the grid on the GPU using the KDTree querying method.

    Args:
    - data_array (numpy.ndarray): Array where each row represents a point with its x, y, z coordinates and features.
    - grid_coords (torch.Tensor): Tensor of shape [batch_size, 3, grid_resolution, grid_resolution].
    - feature_indices (list): List of indices for the selected features.
    - device (torch.device): The device (CPU or GPU) where computations will be performed.

    Returns:
    - feature_grids (torch.Tensor): Tensor of shape [batch_size, num_features, grid_resolution, grid_resolution].
    """
    # Move data_array to the appropriate precision and device
    data_array = data_array.astype(np.float64)
    
    # Create KDTree on CPU
    tree = KDTree(data_array[:, :3])

    batch_size = grid_coords.shape[0]
    grid_resolution = grid_coords.shape[2]
    num_features = len(feature_indices)

    # Initialize a tensor to hold the feature grids
    feature_grids = torch.zeros((batch_size, num_features, grid_resolution, grid_resolution), dtype=torch.float64, device=device)

    # Iterate over each batch element
    for i in range(batch_size):
        # Flatten grid coordinates for querying
        grid_coords_flat = grid_coords[i].permute(1, 2, 0).reshape(-1, 3).cpu().numpy().astype(np.float64)

        # Query KDTree to find nearest point indices
        _, indices = tree.query(grid_coords_flat)

        # Extract features for the nearest points
        nearest_features = data_array[indices][:, feature_indices]  # Correctly extract features using advanced indexing

        # Reshape features to fit the grid and assign to the feature grid tensor
        feature_grids[i] = torch.tensor(nearest_features.reshape(grid_resolution, grid_resolution, num_features).transpose(2, 0, 1), dtype=torch.float64, device=device)

        # Debugging: Print a sample of the grid features
        # print(f"[DEBUG - GPU] Assigned features for batch {i} (sample): {feature_grids[i, :, :5, :5]}")

    return feature_grids

def compare_cpu_gpu_feature_assignment(data_array, x_coords_cpu, y_coords_cpu, constant_z_cpu, grid_coords_gpu, known_features, features_to_use, device):
    """
    Compare feature assignment between the CPU (bulk) and GPU implementations.
    """
    # Determine the feature indices
    feature_indices = [known_features.index(feature) for feature in features_to_use]

    # CPU (bulk) feature assignment
    tree_cpu = KDTree(data_array[:, :3].astype(np.float64))
    grid_cpu = np.zeros((128, 128, len(features_to_use)), dtype=np.float64)  # Assuming grid_resolution=128
    grid_cpu = assign_features_to_grid_bulk(tree_cpu, data_array, grid_cpu, x_coords_cpu, y_coords_cpu, constant_z_cpu, feature_indices)

    # GPU feature assignment using the provided function
    feature_grids_gpu = assign_features_to_grid_gpu(data_array, grid_coords_gpu, feature_indices, device)

    grid_gpu = feature_grids_gpu[0].cpu().numpy()
    print(f'gpu grid shape before transposition:{grid_gpu.shape}')

    grid_gpu_transposed = feature_grids_gpu[0].permute(2, 1, 0)
    print(f'transposed gpu grid shape after transposition:{grid_gpu_transposed.shape}')

    # Convert the GPU grid to numpy for comparison
    grid_gpu_np = grid_gpu_transposed.cpu().numpy()

    # Compare the grids
    if np.allclose(grid_cpu, grid_gpu_np):  # Only compare the first batch for simplicity
        print("[SUCCESS] The grids match between CPU (bulk) and GPU implementations.")
    else:
        print("[ERROR] The grids do not match between CPU (bulk) and GPU implementations.")

    # Visualize the grids to inspect differences
    print("Visualizing CPU grid...")
    grid_cpu = np.transpose(grid_cpu, (2,0,1))
    visualize_grid(grid_cpu, channel=3, save=True, file_path='tests/debug_imgs/CPU', feature_names=feature_names)

    print("Visualizing GPU grid (first batch)...")
    grid_gpu_visual = np.transpose(grid_gpu, (0, 2, 1))
    visualize_grid(grid_gpu_visual, channel=3, save=True, file_path='tests/debug_imgs/GPU', feature_names=feature_names)



#----------------------------------------------------------------------------------------------------------------------------------------------------------

# Load real data from LAS file
las_file_path = 'data/raw/features_F.las'
full_data, feature_names = read_las_file_to_numpy(las_file_path)

features_to_use = ['intensity', 'red', 'green', 'blue']
channels=len(features_to_use)

# Find the indices of the requested features in the known features list
feature_indices = [feature_names.index(feature) for feature in features_to_use]

window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
window_size = 10
grid_resolution=128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tree = KDTree(full_data[:, :3].astype(np.float64))


center_point = full_data[100000, :3]
center_point_tensor = torch.tensor(center_point, dtype=torch.float64).unsqueeze(0)

torch.set_printoptions(precision=8)

gpu_feature_grids, gpu_cell_size, gpu_grid_coords_combined = debug_gpu_create_feature_grid(center_point_tensor, window_size, grid_resolution=128, channels=channels, device=None)

grid_cpu, _, x_coords_cpu, y_coords_cpu, constant_z_cpu = debug_create_feature_grid(center_point, window_size, grid_resolution, channels)

# Inspect the GPU grid's coordinates
x_coords_gpu = gpu_grid_coords_combined[0, 0, :, :]  # X-coordinates for the entire grid (shape: [128, 128])
y_coords_gpu = gpu_grid_coords_combined[0, 1, :, :]  # Y-coordinates for the entire grid (shape: [128, 128])
constant_z_gpu = gpu_grid_coords_combined[0, 2, 0, 0]  # Constant z value for all cells

# Print the coordinates to compare
print(f"[CPU] X-coordinates (sample): {x_coords_cpu[:5]} ... {x_coords_cpu[-5:]}")
print(f"[GPU] X-coordinates (sample): {x_coords_gpu[0, :5].numpy()} ... {x_coords_gpu[0, -5:].numpy()}")

print(f"[CPU] Y-coordinates (sample): {y_coords_cpu[:5]} ... {y_coords_cpu[-5:]}")
print(f"[GPU] Y-coordinates (sample): {y_coords_gpu[:, 0].numpy()[:5]} ... {y_coords_gpu[:, 0].numpy()[-5:]}")

print(f"[CPU] Constant Z: {constant_z_cpu}")
print(f"[GPU] Constant Z: {constant_z_gpu.item()}")

# Inspect the grid shapes
print(f"[CPU] Grid shape: {grid_cpu.shape}")
print(f"[GPU] Grid shape: {gpu_feature_grids[0].shape}")  # Only the first batch's grid

cpu_indices = debug_kd_tree_cpu(full_data, x_coords_cpu, y_coords_cpu, constant_z_cpu)
cpu_bulk_indices = debug_kd_tree_cpu_bulk(full_data, x_coords_cpu, y_coords_cpu, constant_z_cpu)

if cpu_indices[:20] == list(cpu_bulk_indices[:20]):
    print("[SUCCESS] KDTree indices match between CPU and CPU wiht BULK assignment for the sample!")
else:
    print("[ERROR] KDTree indices do not match between CPU and CPU wiht BULK assignment.")

gpu_indices = debug_kd_tree_gpu(full_data, gpu_grid_coords_combined, device)

# Here we only compare the indices
if cpu_indices[:20] == list(gpu_indices[:20]):
    print("[SUCCESS] KDTree indices match between CPU and GPU for the sample!")
else:
    print("[ERROR] KDTree indices do not match between CPU and GPU.")

print('----------------------------KD TREE CHECKS ENDED------------------------------------')

compare_feature_assignments(tree, full_data, x_coords_cpu, y_coords_cpu, constant_z_cpu, feature_indices, grid_resolution)

compare_cpu_gpu_feature_assignment(full_data, x_coords_cpu, y_coords_cpu, constant_z_cpu, gpu_grid_coords_combined, feature_names, features_to_use, device)



"""# Create grids using debug functions
grid_cpu, _, x_coords_cpu, y_coords_cpu, constant_z_cpu = debug_create_feature_grid(center_point, window_size, grid_resolution, channels)
grid_gpu, _, x_coords_gpu, y_coords_gpu, constant_z_gpu = debug_gpu_create_feature_grid(center_point_tensor, window_size, grid_resolution, channels, device)"""


"""# test_single_point_grid(data_array=full_data, center_point=center_point, window_sizes=window_sizes, grid_resolution=grid_resolution, features_to_use=features_to_use, known_features=feature_names)
# test_single_point_grid_gpu(full_data, center_point, window_sizes, grid_resolution, features_to_use, known_features=feature_names, device=device)

# Use the debug functions instead of the originals
grid_cpu, cell_size_cpu, x_coords_cpu, y_coords_cpu, constant_z_cpu = debug_create_feature_grid(center_point, window_size, grid_resolution, channels)
grid_gpu, cell_size_gpu, x_coords_gpu, y_coords_gpu, constant_z_gpu = debug_gpu_create_feature_grid(center_point_tensor, window_size, grid_resolution, channels, device)

# Assign features using the debug versions
grid_cpu = debug_assign_features_to_grid(tree, full_data, grid_cpu, x_coords_cpu, y_coords_cpu, constant_z_cpu, feature_indices)
grid_cpu = np.transpose(grid_cpu, (2, 0, 1))
grid_gpu = debug_gpu_assign_features_to_grid(center_point_tensor, grid_gpu, x_coords_gpu, y_coords_gpu, constant_z_gpu, full_data, tree, feature_indices, device)

visualize_grid(grid=grid_gpu[0])
visualize_grid(grid_cpu)
"""


