import laspy
import numpy as np
import pandas as pd
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


def read_feature_las_files(las_directory='data/raw', feature_suffix='_F', features_to_extract=None, save_to_csv=False,
                           csv_output_dir='data/csv_files'):
    """
    Reads all LAS files with a given suffix in a directory, extracts the scaled coordinate data (x, y, z) and specific features,
    and returns them as a list of pandas DataFrames. Optionally saves the DataFrames as CSV files.

    Parameters:
    - las_directory (str): The directory where the LAS files are stored.
    - feature_suffix (str): The suffix used to identify feature LAS files.
    - features_to_extract (list): List of features to extract from each LAS file. If None, default features will be extracted.
    - save_to_csv (bool): If True, saves the extracted DataFrames to CSV files.
    - csv_output_dir (str): Directory to save the CSV files if save_to_csv is True.

    Returns:
    - List[pd.DataFrame]: A list of pandas DataFrames containing the extracted data from each LAS file.
    """
    # Default features to extract if none are provided
    if features_to_extract is None:
        features_to_extract = ['intensity', 'return_number', 'number_of_returns',
                               'red', 'green', 'blue', 'nir',
                               'ndvi', 'ndwi', 'ssi',
                               'l1_b', 'l2_b', 'l3_b', 'planarity_b',
                               'sphericity_b', 'linearity_b', 'entropy_b', 'theta_b', 'theta_variance_b',
                               'mad_b', 'delta_z_b', 'N_h', 'delta_z_fl'
                               ]

    # Initialize a list to store DataFrames
    dataframes = []

    # Ensure CSV output directory exists if saving to CSV
    if save_to_csv:
        os.makedirs(csv_output_dir, exist_ok=True)

    # Get a list of all LAS files with the specified suffix in the directory
    las_files = glob.glob(os.path.join(las_directory, f'*{feature_suffix}.las'))

    # Iterate over each LAS file
    for las_file in las_files:
        print(f"Processing {las_file}...")

        # Read the LAS file
        las_data = laspy.read(las_file)

        # Extract scaled coordinates (x, y, z) "manually" (to avoid laspy erorrs) and the specified features
        try:
            data = {
                'x': las_data.x,  # Scaled x
                'y': las_data.y,  # Scaled y
                'z': las_data.z  # Scaled z
            }

            # Extract additional features
            for feature in features_to_extract:
                if feature in las_data.point_format.dimension_names:
                    data[feature] = las_data[feature]
                else:
                    print(f"Feature '{feature}' is not available in {las_file}.")

            # Convert the extracted data to a DataFrame
            df = pd.DataFrame(data)
            dataframes.append(df)
            print(f"Loaded DataFrame with shape: {df.shape}")

            # Save to CSV if required
            if save_to_csv:
                print(f"Saving to CSV file...")
                csv_filename = os.path.splitext(os.path.basename(las_file))[0] + '.csv'
                csv_path = os.path.join(csv_output_dir, csv_filename)
                df.to_csv(csv_path, index=False, compression='gzip')
                print(f"Saved DataFrame to CSV: {csv_path} with gzip compression")

        except Exception as e:
            print(f"Error processing file {las_file}: {e}")

    return dataframes


def convert_dataframe_to_numpy(df, selected_features=None):
    """
    Converts the feature DataFrame into a NumPy array with selected features.

    Args:
    - df (pd.DataFrame): DataFrame containing all points and their features.
    - selected_features (list): List of column names or indices to include in the output (default is None, which includes all features).

    Returns:
    - data_array (numpy.ndarray): A NumPy array where each row represents a point and selected features.
    """
    # Select specific features or use all if none provided
    if selected_features:
        df = df[['x', 'y', 'z'] + selected_features]
    else:
        df = df[['x', 'y', 'z'] + [col for col in df.columns if col not in ['x', 'y', 'z']]]

    # Convert the DataFrame to a NumPy array
    data_array = df.to_numpy()
    return data_array


def sample_df(df, sample_size=None, fraction=None, random_state=None):
    """
    Samples a DataFrame either by a fixed number of points or a fraction of the DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame to sample from.
    - sample_size (int, optional): Number of points to sample. If None, use `fraction` instead.
    - fraction (float, optional): Fraction of the DataFrame to sample (between 0.0 and 1.0). Ignored if `sample_size` is specified.
    - random_state (int, optional): Seed for the random number generator for reproducibility.

    Returns:
    - pd.DataFrame: A sampled DataFrame.
    """
    if sample_size is not None:
        # Sample a fixed number of points
        sampled_df = df.sample(n=sample_size, random_state=random_state)
    elif fraction is not None:
        # Sample a fraction of the DataFrame
        sampled_df = df.sample(frac=fraction, random_state=random_state)
    else:
        raise ValueError("Either `sample_size` or `fraction` must be provided.")

    return sampled_df
