import laspy
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import csv


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
    - feature_names (list of str): List of feature names corresponding to the columns in the array.
    """
    # Set default features if none are provided
    if features_to_extract is None:
        features_to_extract = ['intensity', 'return_number', 'red', 'green', 'blue', 'nir',
                               'ndvi', 'ndwi', 'ssi', 'l1_b', 'l2_b', 'l3_b', 'planarity_b', 'sphericity_b',
                               'linearity_b', 'entropy_b', 'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b', 'N_h',
                               'delta_z_fl']

    # Read the LAS file
    print(f"Processing {file_path}...")
    las_data = laspy.read(file_path)

    # Initialize a list to store the features and their names
    data = []
    feature_names = []

    # Check if x, y, z coordinates are present and not empty
    if hasattr(las_data, 'x') and hasattr(las_data, 'y') and hasattr(las_data, 'z'):
        if len(las_data.x) > 0 and len(las_data.y) > 0 and len(las_data.z) > 0:
            # Add x, y, z as the first columns
            data.append(las_data.x)
            feature_names.append('x')
            data.append(las_data.y)
            feature_names.append('y')
            data.append(las_data.z)
            feature_names.append('z')
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
            feature_names.append(feature)
        else:
            print(f"Feature '{feature}' is not available in {file_path}.")

    # Check for segment_id and label fields
    if 'segment_id' in las_data.point_format.dimension_names:
        data.append(las_data['segment_id'])
        feature_names.append('segment_id')

    if 'label' in las_data.point_format.dimension_names:
        data.append(las_data['label'])
        feature_names.append('label')

    # Convert the data list to a numpy array and transpose to match the expected shape (N, num_features)
    data_array = np.vstack(data).T
    print(f"Loaded NumPy array with shape: {data_array.shape}")

    return data_array, feature_names


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
        feature_names = ['x', 'y', 'z', 'intensity', 'number_of_returns',
                         'red', 'green', 'blue', 'nir', 'ndvi', 'ndwi', 'ssi', 'l1_b', 'l2_b',
                         'l3_b', 'planarity_b', 'sphericity_b', 'linearity_b', 'entropy_b',
                         'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b', 'N_h', 'delta_z_fl']

    # Convert the numpy array to a pandas DataFrame
    return pd.DataFrame(data_array, columns=feature_names)


def read_csv_file_to_numpy(file_path, features_to_extract=None):
    """
    Reads a CSV file and extracts the specified features along with coordinates.

    Args:
    - file_path (str): Path to the CSV file.
    - features_to_extract (list of str): List of feature names to extract. 
                                         If None, default features will be extracted.

    Returns:
    - np.ndarray: Numpy array containing the extracted features and coordinates.
    - feature_names (list of str): List of feature names corresponding to the columns in the array.
    """
    # Set default features if none are provided
    if features_to_extract is None:
        features_to_extract = ['intensity', 'return_number', 'red', 'green', 'blue', 'nir',
                               'ndvi', 'ndwi', 'ssi', 'l1_b', 'l2_b', 'l3_b', 'planarity_b', 'sphericity_b',
                               'linearity_b', 'entropy_b', 'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b', 'N_h',
                               'delta_z_fl']

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Ensure 'x', 'y', 'z' coordinates are present
    if not all(coord in df.columns for coord in ['x', 'y', 'z']):
        raise ValueError(f"CSV file {file_path} is missing required coordinates ('x', 'y', 'z').")

    # Extract the coordinates
    coords = df[['x', 'y', 'z']].values

    # Extract the features, checking if they exist in the CSV
    available_features = [f for f in features_to_extract if f in df.columns]
    if not available_features:
        raise ValueError(f"No valid features found in {file_path} from the provided list.")

    feature_data = df[available_features].values

    # Combine coordinates and selected features into a single array
    combined_data = np.hstack((coords, feature_data))
    
    # Combine 'x', 'y', 'z' with the feature names to keep track of the columns
    feature_names = ['x', 'y', 'z'] + available_features

    return combined_data, feature_names


def read_file_to_numpy(data_dir, features_to_use=None, features_file_path=None):
    """
    Loads the raw data from a .npy, .las, or .csv file and returns the data array along with the known features.

    Args:
    - data_dir (str): Path to the raw data file.
    - features_to_use (list): List of features to extract (only used for .las files).
    - features_file_path (str): Path to the features file (only used for .npy files).

    Returns:
    - data_array (np.ndarray): The raw data array (x, y, z, features).
    - known_features (list): List of feature names corresponding to the data array.
    """
    if data_dir.endswith('.npy'):  # Directly using a NumPy array as raw data
        print("Loading raw data from numpy file...")
        data_array = np.load(data_dir)
        try:
            known_features = load_features_used(features_file_path)
            print(f"Features loaded from {features_file_path}: {known_features}")
        except Exception as e:
            raise ValueError(f"Unable to load features from {features_file_path}: {e}")

    elif data_dir.endswith('.las'):  # Raw LAS file
        print("Loading raw data from LAS file...")
        data_array, known_features = read_las_file_to_numpy(data_dir, features_to_extract=features_to_use)

    elif data_dir.endswith('.csv'):  # Raw CSV file
        print("Loading raw data from CSV file...")
        df = pd.read_csv(data_dir)
        data_array = df.values
        known_features = df.columns.tolist()

    else:
        raise ValueError("Unsupported data format. Please provide a .npy, .las, or .csv file.")

    return data_array, known_features



def combine_and_save_csv_files(csv_files, save=False, save_dir='data/combined_data'):
    """
    Combines multiple CSV files into a single NumPy array and optionally saves the combined data to a file.

    Args:
    - csv_files (list of str): List of paths to CSV files.
    - save (bool): Whether to save the combined data to a file. Default is False.
    - save_dir (str): Directory where the combined NumPy array will be saved. Default is 'combined_data'.

    Returns:
    - np.ndarray: Combined data from all CSV files as a NumPy array.
    """
    combined_data = []

    # Loop through each CSV file and read its contents
    print("Reading CSV files:")
    for file in tqdm(csv_files, desc="Reading", unit="file"):
        # Read the CSV file into a NumPy array
        data = pd.read_csv(file).values
        combined_data.append(data)

    # Combine all data into a single NumPy array
    combined_array = np.vstack(combined_data)

    # Optionally save the combined data
    if save:
        os.makedirs(save_dir, exist_ok=True)
        output_file_path = os.path.join(save_dir, 'combined_data.npy')
        np.save(output_file_path, combined_array)
        print(f"Combined data saved to {output_file_path}")

    return combined_array


def sample_data(input_file, sample_size, save=False, save_dir='data/sampled_data', feature_to_use=None, features_file_path=None):
    """
    Samples a subset of the data from a CSV, NumPy, or LAS file and saves it as a CSV with metadata.

    Args:
    - input_file (str): Path to the input file (either a CSV, NumPy file, or LAS file).
    - sample_size (int): The number of samples to extract.
    - file_type (str): The type of the input file ('csv', 'npy', or 'las').
    - save (bool): Whether to save the sampled data to a file. Default is False.
    - save_dir (str): Directory where the sampled data will be saved. Default is 'data/sampled_data'.
    - feature_to_use (list): List of feature names to select from the data.
    - features_file_path (str): File path to known features of the .npy data (required only if input is NumPy).

    Returns:
    - np.ndarray: The sampled subset of the data.
    """
    # Load the data 
    data_array, feature_names = read_file_to_numpy(data_dir=input_file, features_to_use=feature_to_use, features_file_path=features_file_path)

    # Check if sample_size is greater than the dataset size
    if sample_size > data_array.shape[0]:
        raise ValueError(f"Sample size {sample_size} is larger than the dataset size {data_array.shape[0]}.")

    # Sample the data
    print(f"Sampling {sample_size} rows from the dataset...")
    sampled_data = data_array[np.random.choice(data_array.shape[0], sample_size, replace=False)]

    # Optionally save the sampled data as a CSV
    if save:
        os.makedirs(save_dir, exist_ok=True)
        output_file_path = os.path.join(save_dir, f'sampled_data_{sample_size}.csv')
        df_sampled = pd.DataFrame(sampled_data, columns=feature_names)  # Convert to DataFrame with column names
        df_sampled.to_csv(output_file_path, index=False)
        print(f"Sampled data saved to {output_file_path}")

    return sampled_data


def save_features_used(features_to_use, save_dir):
    """
    Saves the features used during grid generation to a CSV file.

    Args:
    - features_to_use (list): List of features used in grid generation.
    - save_dir (str): Directory where the grids are saved.

    Returns:
    - None
    """
    feature_file = os.path.join(save_dir, 'features_used.csv')
    with open(feature_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(features_to_use)
    print(f"Features saved: {features_to_use}")


def load_features_used(features_file_path):
    """
    Loads the features used for grid generation from a saved CSV file.

    Args:
    - features_file_path (str): Path to the CSV file containing the saved features.

    Returns:
    - features_list (list): List of features loaded from the CSV file.

    Raises:
    - FileNotFoundError: If the features file is not found.
    - ValueError: If the file is empty or not in the expected format.
    """

    # Check if the file exists
    if not os.path.exists(features_file_path):
        raise FileNotFoundError(f"Features file not found at {features_file_path}")

    # Load the features from the CSV file
    try:
        with open(features_file_path, 'r') as f:
            reader = csv.reader(f)
            features_list = next(reader)  # Assuming the first row contains the feature names

        if not features_list:
            raise ValueError(f"The features file at {features_file_path} is empty or invalid.")

        return features_list

    except Exception as e:
        raise ValueError(f"Error loading features from {features_file_path}: {e}")


def remap_labels(data_array, label_column_index=-1):
    """
    Automatically remaps the labels in the given data array to a continuous range starting from 0. Needed for
    training purpose (in order to feed labels as targets to the loss).
    Stores the mapping for future reference.

    Args:
    - data_array (np.ndarray): The input data array where the last column (by default) contains the labels.
    - label_column_index (int): The index of the column containing the labels (default is the last column).

    Returns:
    - np.ndarray: The data array with the labels remapped.
    - dict: A dictionary that stores the original to new label mapping.
    """
    # Extract the label column
    labels = data_array[:, label_column_index]

    # Get the unique labels
    unique_labels = np.unique(labels)

    # Create a mapping from the unique labels to continuous integers
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    # Apply the mapping to the labels
    remapped_labels = np.array([label_mapping[label] for label in labels])

    # Replace the original labels in the data array with the remapped labels
    data_array[:, label_column_index] = remapped_labels

    # Return the remapped data array and the mapping dictionary
    return data_array, label_mapping


def extract_num_classes(pre_process_data, preprocessed_data_dir=None, raw_file_path=None):
    """
    Extracts the number of unique classes from raw data (LAS, CSV, or NPY) or from preprocessed labels.

    Args:
    - raw_file_path (str): Path to the input LAS, CSV, or NPY file.
    - pre_process_data (bool): If True, extract classes from raw data; if False, extract classes from preprocessed files.
    - preprocessed_data_dir (str, optional): Path to the preprocessed data directory (required if pre_process_data=False).

    Returns:
    - int: The number of unique classes.
    """

    if pre_process_data:
        # Load data from raw files
        data_array, _ = read_file_to_numpy(raw_file_path)
        
        # Extract class labels from the last column of the data array
        class_labels = data_array[:, -1]

        # Extract the unique number of classes
        num_classes = len(np.unique(class_labels))
        
    else: 
        if preprocessed_data_dir is None:
            raise ValueError("preprocessed_data_dir is required when pre_process_data is False.")

        # Load the labels from the unified 'labels.csv' file
        labels_file = os.path.join(preprocessed_data_dir, 'labels.csv')
        
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found in {labels_file}")
        
        # Read the labels CSV and extract the class labels
        labels_df = pd.read_csv(labels_file)
        class_labels = labels_df['label'].values

        # Extract the unique number of classes from the labels
        num_classes = len(np.unique(class_labels))

    print(f"Number of unique classes: {num_classes}")
    
    return num_classes


def extract_num_channels(preprocessed_data_dir):
    """
    Extracts the number of channels from the preprocessed grid files.

    Args:
    - preprocessed_data_dir (str): Path to the preprocessed data directory.

    Returns:
    - int: The number of channels in the preprocessed grids.
    """
    # Assuming grids are stored in the 'small' directory (or any scale's directory)
    grid_files = os.listdir(os.path.join(preprocessed_data_dir, 'small'))
    
    if not grid_files:
        raise FileNotFoundError(f"No grid files found in the directory: {os.path.join(preprocessed_data_dir, 'small')}")

    # Load one grid file to check its shape
    sample_grid = np.load(os.path.join(preprocessed_data_dir, 'small', grid_files[0]))
    
    # The number of channels is the first dimension in the shape (assuming format: [channels, height, width])
    num_channels = sample_grid.shape[0]
    
    return num_channels