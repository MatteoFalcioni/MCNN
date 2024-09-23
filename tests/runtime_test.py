import time
import cProfile
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LinearRegression
from data.transforms.point_cloud_to_image import generate_grids_for_training


def test_total_grid_generation_runtime(num_points_list, window_sizes, grid_resolution=128, channels=10,
                                       save_to_file=True):
    """
    Tests and tracks total runtime for generating grids at multiple window sizes for different point cloud sizes.
    Visualizes the result and saves the data to a CSV file if save_to_file is True.
    """
    timings = []  # To store total timings for each point cloud size

    # Optional CSV file to store the results
    if save_to_file:
        output_file = 'tests/test_runtime/grid_generation_total_runtime_data.csv'
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Number of Points', 'Total Time Taken (seconds)'])

            for num_points in num_points_list:
                # Generate random point cloud (x, y, z, features)
                data_array = np.random.rand(num_points, channels + 3)

                # Track start time for generating grids for all window sizes
                start_time = time.time()

                # Generate grids for each window size (3 total grids per point set)
                for ws in window_sizes:
                    generate_grids_for_training(data_array, ws, grid_resolution, channels)

                # Track end time and calculate total elapsed time
                end_time = time.time()
                total_elapsed_time = end_time - start_time
                timings.append(total_elapsed_time)

                print(f"Total time for {num_points} points: {total_elapsed_time:.2f} seconds")

                # Save the total time data to CSV
                writer.writerow([num_points, total_elapsed_time])

    # Plot the total timings for all point cloud sizes
    plt.figure(figsize=(10, 6))
    plt.plot(num_points_list, timings, label="Total Time for All Window Sizes")

    plt.xlabel('Number of Points')
    plt.ylabel('Total Time (seconds)')
    plt.title('Total Grid Generation Time by Point Cloud Size')
    plt.legend()
    plt.grid(True)

    # Save it to file
    file_path = 'tests/test_runtime/total_grid_generation_time_complexity.png'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path)

    plt.close()


"""# Set of different window sizes to test
window_sizes = [2.5, 5, 10]

# List of increasing number of points to test
num_points_list = [50, 100, 200, 300, 500, 1000, 2000]

# Run the profiling and visualize the time
cProfile.run('test_total_grid_generation_runtime(num_points_list, window_sizes)')"""


def fit_and_predict_runtime_from_csv(csv_file, prediction_point=100000):
    """
    Fits a linear model to the grid generation runtime data from a CSV file and predicts
    the runtime for a larger number of points.

    Args:
    - csv_file (str): Path to the CSV file with point cloud sizes and runtime data.
    - prediction_point (int): The number of points to predict runtime for.

    Returns:
    - predicted_time (float): Predicted runtime for the given number of points.
    """
    # Load data from the CSV

    num_points_list = []
    timings = []

    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        for row in reader:
            num_points_list.append(int(row[0]))
            timings.append(float(row[1]) / 60)  # Convert to minutes

    # Convert lists to numpy arrays for linear regression
    X = np.array(num_points_list).reshape(-1, 1)
    y = np.array(timings)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the runtime for the given prediction point
    predicted_time = model.predict(np.array([[prediction_point]]))

    print(f"Predicted runtime for {prediction_point} points: {predicted_time[0]:.2f} minutes")

    # Plot the data and the linear fit
    plt.figure(figsize=(10, 6))
    plt.plot(num_points_list, timings, label="Measured Total Time", marker='o')
    plt.plot([0, prediction_point], model.predict(np.array([[0], [prediction_point]])),
             label=f"Linear Fit (predicted for {prediction_point} points)", linestyle="--")
    plt.xlabel('Number of Points')
    plt.ylabel('Total Time (minutes)')
    plt.title('Total Grid Generation Time with Linear Fit, on CPU')
    plt.legend()
    plt.grid(True)

    plt.savefig('tests/test_runtime/runtime_linear_regression')
    plt.show()
    plt.close()

    return predicted_time[0]


csv_file_path = 'tests/test_runtime/grid_generation_total_runtime_data.csv'
predicted_runtime = fit_and_predict_runtime_from_csv(csv_file_path)


