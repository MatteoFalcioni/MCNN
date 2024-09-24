import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np


def plot_loss(train_losses, val_losses, save_dir='results/plots/'):
    """
    Plots the training and validation loss over epochs and saves the plot with a label for the model.

    Args:
    - train_losses (list of float): List containing the training loss values for each epoch.
    - val_losses (list of float): List containing the validation loss values for each epoch.
    - save_dir (str): Directory where to save the plot. Defaults to 'results/plots/'.
    """
    # Create the 'plots' folder inside the results directory
    os.makedirs(save_dir, exist_ok=True)

    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the path to save the plot
    plot_path = os.path.join(save_dir, f"MCNN_loss_plot_{timestamp}.png")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linestyle='-', marker='o', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linestyle='-', marker='o', color='orange', linewidth=2)
    plt.title(f'Training and Validation Loss for MCNN', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    # Save the plot to the specified path
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # Close the plot
    plt.close()


def visualize_grid(grid, channel=0, title="Grid Visualization", save=False, file_path=None):
    """
    Visualizes a specific channel of the grid, which is expected to be in PyTorch format (channels, H, W).

    Args:
    - grid (numpy.ndarray): A 3D grid array with shape (channels, grid_resolution, grid_resolution).
    - channel (int): The channel to visualize (default is 0 for the first channel).
    - title (str): Title for the plot.
    - save (bool): If True, saves the plot to a file instead of showing it.
    - file_path (str): File path to save the plot if save is True.

    Returns:
    - None: Displays the plot.
    """
    if grid.ndim != 3:
        raise ValueError("Grid must be a 3D array with shape (channels, grid_resolution, grid_resolution).")

    if channel >= grid.shape[0]:
        raise ValueError(f"Channel {channel} is out of bounds for this grid with {grid.shape[0]} channels.")

    # Extract the specified channel in the (H, W) format
    grid_channel = grid[channel, :, :]

    # Plotting the grid using a heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_channel, cmap='viridis', interpolation='nearest')
    plt.colorbar(label=f'Feature Value (Channel {channel})')
    plt.title(title)
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')

    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if save and file_path:
        plt.savefig(file_path)
        plt.show()
        plt.close()
    elif save and not file_path:
        print("No saving path added for the plot. Saving disabled.")
        plt.show()
    elif not save:
        plt.show()

def visualize_dtm(dtm_data):
    """
    Visualizza il Digital Terrain Model (DTM) con una legenda.

    Args:
    - dtm_data (np.ndarray): Array numpy con i dati del DTM.
    """
    plt.figure(figsize=(10, 8))
    im = plt.imshow(dtm_data, cmap='terrain', interpolation='nearest')

    # Add colorbar with legend for elevation values
    cbar = plt.colorbar(im, orientation='vertical')
    cbar.set_label('Elevation (meters)', rotation=270, labelpad=15)

    plt.title('Digital Terrain Model (DTM)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    plt.show()


def plot_point_cloud_with_rgb(df, save=False, file_path=None):
    """
    Plots a 3D scatter plot of a point cloud with RGB colors.

    Args:
    - df (pandas.DataFrame): DataFrame containing 'x', 'y', 'z', 'red', 'green', 'blue' columns.
    - save (bool): If True, saves the plot to a file instead of showing it.
    - file_path (str): File path to save the plot if save is True.
    """
    # Ensure that RGB values are rescaled (from 16bit to 8bit and then in range 0-1)
    df[['red', 'green', 'blue']] = (df[['red', 'green', 'blue']] / 65535).astype(np.float32)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with RGB colors
    ax.scatter(df['x'], df['y'], df['z'], c=df[['red', 'green', 'blue']].values, s=0.1)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if save and file_path:
        plt.savefig(file_path)
        plt.show()
        plt.close()
    elif save and not file_path:
        print("No saving path added for the plot. Saving disabled.")
        plt.show()
    elif not save:
        plt.show()


def visualize_grid_with_comparison(grid, df, center, window_size=10.0, channel=3, feature_names=None, visual_size=100, save=False, file_path=None):
    """
    Visualize the grid and the filtered point cloud together.

    Args:
    - grid (np.ndarray): The grid to visualize.
    - df (pd.DataFrame): DataFrame of the point cloud.
    - center (tuple): (x, y, z) coordinates of the center point.
    - window_size (float): Size of the window (in meters) around the center point.
    - channel (int): The channel to visualize from the grid (default is 0).
    - feature_names (list): List of feature names corresponding to grid channels.
    - visual_size (float): Size of the visualization area around the center point.
    - save (bool): If True, saves the plot to a file instead of showing it.
    - file_path (str): File path to save the plot if save is True.
    """

    # Define the extents of the grid area and point cloud visualization area
    x_min, x_max = center[0] - (window_size / 2), center[0] + (window_size / 2)
    y_min, y_max = center[1] - (window_size / 2), center[1] + (window_size / 2)

    x_visualize_min, x_visualize_max = center[0] - (visual_size / 2), center[0] + (visual_size / 2)
    y_visualize_min, y_visualize_max = center[1] - (visual_size / 2), center[1] + (visual_size / 2)
    z_visualize_min, z_visualize_max = center[2] - (visual_size / 2), center[2] + (visual_size / 2)

    # Filter the points within the visualization range
    filtered_df = df[(df['x'] >= x_visualize_min) & (df['x'] <= x_visualize_max) &
                     (df['y'] >= y_visualize_min) & (df['y'] <= y_visualize_max) &
                     (df['z'] >= z_visualize_min) & (df['z'] <= z_visualize_max)]

    print("DataFrame has been filtered")

    fig = plt.figure(figsize=(12, 6))

    # Plot the grid
    ax1 = fig.add_subplot(121)
    grid_channel = grid[:, :, channel]
    im = ax1.imshow(grid_channel, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax1, label=f'Feature Value (Channel {channel})')
    feature_name = feature_names[channel] if feature_names and channel < len(feature_names) else f"Channel {channel}"
    ax1.set_title(f'Grid Visualization: {feature_name}')
    ax1.set_xlabel('Grid X')
    ax1.set_ylabel('Grid Y')

    # Plot the filtered point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(filtered_df['x'], filtered_df['y'], filtered_df['z'],
                c=filtered_df[['red', 'green', 'blue']].values / 65535, s=0.1)
    ax2.set_title('Point Cloud Subset Visualization')

    # Draw a red bounding box around the area of the grid on the point cloud
    z_center = filtered_df['z'].mean()  # Calculate mean Z for the bounding box level
    ax2.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min],
             [z_center, z_center, z_center, z_center, z_center], color='r')

    # Optionally plot the center point to verify its location
    ax2.scatter(center[0], center[1], center[2], c='black', s=200, label='Center Point', marker='x')

    z_min, z_max = df['z'].min(), df['z'].max()  # Use full dataset's Z range

    # Define the 8 corners of the parallelepiped
    corners = np.array([[x_min, y_min, z_min],
                        [x_min, y_max, z_min],
                        [x_max, y_max, z_min],
                        [x_max, y_min, z_min],
                        [x_min, y_min, z_max],
                        [x_min, y_max, z_max],
                        [x_max, y_max, z_max],
                        [x_max, y_min, z_max]])

    # List of edges to connect the corners
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),  # Bottom rectangle
             (4, 5), (5, 6), (6, 7), (7, 4),  # Top rectangle
             (0, 4), (1, 5), (2, 6), (3, 7)]  # Vertical edges

    # Draw the edges
    for edge in edges:
        ax2.plot3D(*zip(corners[edge[0]], corners[edge[1]]), color="r",zorder=10)

    plt.legend()

    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # save to file
    if save and file_path:
        plt.savefig(file_path)
        plt.show()
        plt.close()
    elif save and not file_path:
        print("No saving path added for the plot. Saving disabled.")
        plt.show()
    elif not save:
        plt.show()

