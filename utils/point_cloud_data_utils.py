import laspy
import numpy as np
import pandas as pd


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


def read_las_file_to_numpy(file_path, features_to_extract=None):
    """
    Reads a LAS file, extracts coordinate data (x, y, z) and specific features,
    and returns them as a numpy array.

    Parameters:
    - file_path (str): The path to the LAS file.
    - features_to_extract (list): List of features to extract from the LAS file.
                                  If None, default features will be extracted.

    Returns:
    - np.ndarray: A numpy array containing the extracted data from the LAS file.
    """
    # Set default features if none are provided
    if features_to_extract is None:
        features_to_extract = ['intensity', 'return_number', 'number_of_returns', 'red', 'green', 'blue', 'nir',
                               'ndvi', 'ndwi', 'ssi', 'l1_b', 'l2_b', 'l3_b', 'planarity_b', 'sphericity_b',
                               'linearity_b', 'entropy_b', 'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b', 'N_h',
                               'delta_z_fl']

    # Read the LAS file
    print(f"Processing {file_path}...")
    las_data = laspy.read(file_path)

    # Initialize a list to store the features for this file
    data = []

    # Check if x, y, z coordinates are present and not empty
    if hasattr(las_data, 'x') and hasattr(las_data, 'y') and hasattr(las_data, 'z'):
        if len(las_data.x) > 0 and len(las_data.y) > 0 and len(las_data.z) > 0:
            # Add x, y, z as the first columns
            data.append(las_data.x)
            data.append(las_data.y)
            data.append(las_data.z)
        else:
            print(f"Warning: One of the coordinate arrays (x, y, z) is empty in {file_path}.")
            return None
    else:
        print(f"Warning: LAS data in {file_path} does not have 'x', 'y', or 'z' attributes.")
        return None

    # Extract additional features
    for feature in features_to_extract:
        if feature in ['x', 'y', 'z']:
            continue  # Skip if feature is x, y, or z since they are already added
        if feature in las_data.point_format.dimension_names:
            data.append(las_data[feature])
        else:
            print(f"Feature '{feature}' is not available in {file_path}.")

    # Convert the data list to a numpy array and transpose to match the expected shape (N, num_features)
    data_array = np.vstack(data).T
    print(f"Loaded NumPy array with shape: {data_array.shape}")

    return data_array


def numpy_to_dataframe(data_array, feature_names=None):
    """
    Converts a NumPy array to a pandas DataFrame.

    Args:
    - data_array (numpy.ndarray): The NumPy array to convert.
    - feature_names (list): List of column names for the DataFrame.

    Returns:
    - pandas.DataFrame: The resulting DataFrame.
    """

    # Define default feature names if not provided
    if feature_names is None:
        feature_names = ['x', 'y', 'z', 'intensity', 'return_number', 'number_of_returns',
                         'red', 'green', 'blue', 'nir', 'ndvi', 'ndwi', 'ssi', 'l1_b', 'l2_b',
                         'l3_b', 'planarity_b', 'sphericity_b', 'linearity_b', 'entropy_b',
                         'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b', 'N_h', 'delta_z_fl']

    # Convert the numpy array to a pandas DataFrame
    return pd.DataFrame(data_array, columns=feature_names)
