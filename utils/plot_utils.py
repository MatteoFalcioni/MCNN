import matplotlib.pyplot as plt
import os
from datetime import datetime


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


def visualize_grid(grid, channel=0, title="Grid Visualization"):
    """
    Visualizes a specific channel of the grid.

    Args:
    - grid (numpy.ndarray): A 2D grid array with shape (grid_resolution, grid_resolution, channels).
    - channel (int): The channel to visualize (default is 0 for the first channel).
    - title (str): Title for the plot.

    Returns:
    - None: Displays the plot.
    """
    if grid.ndim != 3:
        raise ValueError("Grid must be a 3D array with shape (grid_resolution, grid_resolution, channels).")

    if channel >= grid.shape[2]:
        raise ValueError(f"Channel {channel} is out of bounds for this grid with {grid.shape[2]} channels.")

    # Extract the specified channel
    grid_channel = grid[:, :, channel]

    # Plotting the grid using a heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_channel, cmap='viridis', interpolation='nearest')
    plt.colorbar(label=f'Feature Value (Channel {channel})')
    plt.title(title)
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
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

