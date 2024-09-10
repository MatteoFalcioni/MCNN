import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def load_las_data(file_path):
    """
    Carica il file LAS e restituisce le coordinate XYZ.

    Args:
    - file_path (str): Percorso al file LAS.

    Returns:
    - np.ndarray: Coordinate XYZ dei punti.
    """
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    return points


def load_asc_data(file_path):
    """
    Carica il file ASC (DTM) e restituisce un array numpy.

    Args:
    - file_path (str): Percorso al file ASC.

    Returns:
    - np.ndarray: Dati del DTM.
    """
    dtm_data = np.loadtxt(file_path, skiprows=6)  # Skips metadata lines (i.e., skips header)
    return dtm_data


def read_feature_las_files(las_directory='data/raw', feature_suffix='_F', variables=None, save_to_csv=False,
                           csv_directory='data/csv_files'):
    """
    Reads all LAS files with a specific suffix (e.g., '_F') in a specified directory,
    extracts the relevant features (computed with radius 1m, i.e., ending with '_b'), and
    converts each LAS file into a separate pandas DataFrame. Optionally saves each DataFrame to a CSV file.

    Args:
    - las_directory (str): Path to the directory containing LAS files. Default is 'data/raw'.
    - feature_suffix (str): Suffix to identify feature LAS files. Default is '_F'.
    - variables (list): List of features to extract from LAS files. Default is None, meaning extract all.
    - save_to_csv (bool): If True, saves each DataFrame to a CSV file. Default is False.
    - csv_directory (str): Directory to save the CSV files. Default is 'data/csv_files'.

    Returns:
    - dict: A dictionary where each key is the LAS file name and the value is the corresponding DataFrame.
    """
    # Default features to extract (those ending with '_b' for radius = 1m)
    if variables is None:
        variables = [
            'x', 'y', 'z', 'intensity',  # Common point features
            'ndvi', 'ndwi', 'ssi', 'N_h', 'delta_z_fl',  # Other non-radius features
            # Features computed with radius 1m
            'l1_b', 'l2_b', 'l3_b', 'planarity_b', 'sphericity_b', 'linearity_b',
            'entropy_b', 'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b'
        ]

    # Get a list of all feature LAS files in the directory
    feature_files = glob.glob(os.path.join(las_directory, f'*{feature_suffix}.las'))

    # Dictionary to store DataFrames for each LAS file
    las_dataframes = {}

    # Create directory for CSV files if saving is enabled
    if save_to_csv:
        os.makedirs(csv_directory, exist_ok=True)

    # Iterate over each feature LAS file
    for las_file in feature_files:
        print(f"Processing {las_file}...")

        # Read the LAS file and extract the data
        las_data = laspy.read(las_file)

        # Create a DataFrame from the LAS data for the specified features
        df = pd.DataFrame(
            {var: np.array(las_data[var]) for var in variables if var in las_data.point_format.dimension_names})

        # Store the DataFrame in the dictionary
        las_dataframes[os.path.basename(las_file)] = df

        # Save to CSV if needed
        if save_to_csv:
            csv_path = os.path.join(csv_directory, f"{os.path.basename(las_file).replace('.las', '.csv')}")
            df.to_csv(csv_path, index=False)
            print(f"Saved {csv_path}")

    return las_dataframes


def convert_dataframe_to_numpy(features_df, selected_features=None):
    """
    Converts the feature DataFrame into a NumPy array with selected features.

    Args:
    - features_df (pd.DataFrame): DataFrame containing all points and their features.
    - selected_features (list): List of column names or indices to include in the output (default is None, which includes all features).

    Returns:
    - data_array (numpy.ndarray): A NumPy array where each row represents a point and selected features.
    """
    if selected_features is not None:
        # Filter the DataFrame to include only the selected features
        filtered_df = features_df[['x', 'y', 'z'] + selected_features]
    else:
        # Convert entire DataFrame to a NumPy array if no selection is made
        filtered_df = features_df

    # Convert the DataFrame to a NumPy array
    data_array = filtered_df.to_numpy()
    return data_array




