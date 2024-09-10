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
                           csv_output_dir='data/csv_files', sample_size=None, sample_fraction=None, random_state=42):
    """
    Reads all LAS files with a given suffix in a directory, extracts the coordinate data (x, y, z) and specific features,
    and returns them as a list of pandas DataFrames. Optionally saves the DataFrames as CSV files and allows sampling of the data.

    Parameters:
    - las_directory (str): The directory where the LAS files are stored.
    - feature_suffix (str): The suffix used to identify feature LAS files.
    - features_to_extract (list): List of features to extract from each LAS file. If None, default features will be extracted.
    - save_to_csv (bool): If True, saves the extracted DataFrames to CSV files.
    - csv_output_dir (str): Directory to save the CSV files if save_to_csv is True.
    - sample_size (int, optional): Number of points to sample. If None, use `sample_fraction` instead.
    - sample_fraction (float, optional): Fraction of the DataFrame to sample (between 0.0 and 1.0). Ignored if `sample_size` is specified.
    - random_state (int, optional): Seed for the random number generator for reproducibility.

    Returns:
    - List[pd.DataFrame]: A list of pandas DataFrames containing the extracted data from each LAS file.
    """
    if features_to_extract is None:
        features_to_extract = ['intensity', 'return_number', 'number_of_returns', 'red', 'green', 'blue', 'nir',
                               'ndvi', 'ndwi', 'ssi', 'l1_b', 'l2_b', 'l3_b', 'planarity_b', 'sphericity_b',
                               'linearity_b', 'entropy_b', 'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b', 'N_h', 'delta_z_fl']

    dataframes = []

    if save_to_csv:
        os.makedirs(csv_output_dir, exist_ok=True)

    las_files = glob.glob(os.path.join(las_directory, f'*{feature_suffix}.las'))

    for las_file in las_files:
        print(f"Processing {las_file}...")
        las_data = laspy.read(las_file)

        try:
            # Initialize data dictionary
            data = {}

            if hasattr(las_data, 'x') and hasattr(las_data, 'y') and hasattr(las_data, 'z'):
                if len(las_data.x) > 0 and len(las_data.y) > 0 and len(las_data.z) > 0:
                    data['x'] = las_data.x  # Scaled x
                    data['y'] = las_data.y  # Scaled y
                    data['z'] = las_data.z  # Scaled z
                else:
                    print(f"Warning: One of the coordinate arrays (x, y, z) is empty in {las_file}.")
            else:
                print(f"Warning: LAS data in {las_file} does not have 'x', 'y', or 'z' attributes.")

            # Extract additional features, skipping x, y, z
            for feature in features_to_extract:
                if feature in ['x', 'y', 'z']:
                    continue  # Skip if feature is x, y, or z since they are already added
                if feature in las_data.point_format.dimension_names:
                    data[feature] = las_data[feature]
                else:
                    print(f"Feature '{feature}' is not available in {las_file}.")

            df = pd.DataFrame(data)

            # Apply sampling if needed. Get a smaller sampled dataframe
            if sample_size is not None:
                df = df.sample(n=sample_size, random_state=random_state)
                print(f"Sampled DataFrame with {sample_size} points: {df.shape}")
            elif sample_fraction is not None:
                df = df.sample(frac=sample_fraction, random_state=random_state)
                print(f"Sampled DataFrame with fraction {sample_fraction}: {df.shape}")

            dataframes.append(df)
            print(f"Loaded DataFrame with shape: {df.shape}")

            if save_to_csv:
                print(f"Saving to CSV file...")
                csv_filename = os.path.splitext(os.path.basename(las_file))[0] + '.csv.gz'
                csv_path = os.path.join(csv_output_dir, csv_filename)
                df.to_csv(csv_path, index=False, compression='gzip')
                print(f"Saved DataFrame to CSV: {csv_path} with gzip compression")

        except Exception as e:
            print(f"Error processing file {las_file}: {e}")

    return dataframes

