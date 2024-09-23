import torch
import numpy as np
from scipy.spatial import KDTree

# Simulate data for testing (replace this with your actual data loading process)
batch_size = 2
num_points = 100
num_features = 3
channels = 3
grid_resolution = 128
window_size = 10.0

# Simulated point cloud data: [batch_size, num_points, 3] (x, y, z)
batch_data = torch.rand((batch_size, num_points, 3))

# Simulated feature data: [batch_size, num_points, num_features]
batch_features = torch.rand((batch_size, num_points, num_features))

# Device setup (CPU for now, switch to 'cuda' for GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move data to device
batch_data = batch_data.to(device)
batch_features = batch_features.to(device)

# Verbose output: initial data
print("Initial batch_data:", batch_data)
print("Initial batch_features:", batch_features)

# Step 1: Grid creation

# Calculate the size of each cell in meters
cell_size = window_size / grid_resolution
print(f"Cell size: {cell_size}")

# Initialize the grids with zeros: [batch_size, channels, grid_resolution, grid_resolution]
grids = torch.zeros((batch_size, channels, grid_resolution, grid_resolution), device=device)
print(f"Initialized grids with shape: {grids.shape}")

# Calculate half the resolution plus 0.5 to center the grid coordinates
half_resolution_plus_half = (grid_resolution / 2) + 0.5
print(f"Half resolution plus 0.5: {half_resolution_plus_half}")

# Create x and y coordinate grids for each point in the batch
x_coords = batch_data[:, :, 0].unsqueeze(2) - (
    half_resolution_plus_half - torch.arange(grid_resolution, device=device).view(1, 1, -1)) * cell_size
y_coords = batch_data[:, :, 1].unsqueeze(2) - (
    half_resolution_plus_half - torch.arange(grid_resolution, device=device).view(1, 1, -1)) * cell_size

# Verbose output: x and y coordinates
print("x_coords sample (batch 0):", x_coords[0, :5, :5])  # Show first 5 points and first 5 coords
print("y_coords sample (batch 0):", y_coords[0, :5, :5])

# Check grid shapes
print("Grids shape:", grids.shape)
print("x_coords shape:", x_coords.shape)
print("y_coords shape:", y_coords.shape)

# Step 2: Feature assignment

# Flatten grid coordinates for KDTree query
flat_x_coords = x_coords.view(batch_size, -1)
flat_y_coords = y_coords.view(batch_size, -1)

# KDTree building and querying
for i in range(batch_size):
    # Extract the (x, y) points for the current batch
    points_cpu = batch_data[i, :, :2].cpu().numpy()  # Convert to NumPy for KDTree
    print(f"Points (batch {i}) shape:", points_cpu.shape)
    print(f"Points (batch {i}) sample:", points_cpu[:5])  # Print first 5 points for debugging

    tree = KDTree(points_cpu)  # Build KDTree for this batch
    print(f"KDTree built for batch {i}")

    # Flattened coordinates for KDTree query
    flat_coords = torch.stack([flat_x_coords[i], flat_y_coords[i]], dim=1).cpu().numpy()
    print(f"Flattened coords (batch {i}) shape:", flat_coords.shape)
    print(f"Flattened coords (batch {i}) sample:", flat_coords[:5])  # First 5 flattened coords

    # Query KDTree
    _, idxs = tree.query(flat_coords)
    print(f"KDTree query indices shape (batch {i}):", idxs.shape)
    print(f"KDTree query indices sample (batch {i}):", idxs[:5])  # First 5 indices

    # Assign features based on KDTree results
    grids[i] = batch_features[i, idxs].T  # Assign features to grid
    print(f"Grids updated for batch {i}")

# Check grids after assignment
print("Grids after feature assignment:", grids)

# Optional: Visualize the grids for debugging purposes
for i in range(batch_size):
    grid_np = grids[i].cpu().numpy()
    print(f"Grid for batch {i} (first channel): {grid_np[0]}")
