import laspy
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


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


def sample_data(input_file, sample_size, file_type='csv', save=False, save_dir='data/sampled_data', feature_names=None):
    """
    Samples a subset of the data from a CSV, NumPy, or LAS file and saves it as a CSV with metadata.

    Args:
    - input_file (str): Path to the input file (either a CSV, NumPy file, or LAS file).
    - sample_size (int): The number of samples to extract.
    - file_type (str): The type of the input file ('csv', 'npy', or 'las').
    - save (bool): Whether to save the sampled data to a file. Default is False.
    - save_dir (str): Directory where the sampled data will be saved. Default is 'data/sampled_data'.
    - feature_names (list): List of feature names for the data (required if input is NumPy).

    Returns:
    - np.ndarray: The sampled subset of the data.
    """
    # Load the data based on file type
    if file_type == 'csv':
        print("Reading data from CSV file...")
        data = []
        for file in tqdm(input_file, desc="Loading CSV files", unit="file"):
            df = pd.read_csv(file)
            data.append(df)
        data = pd.concat(data)
        feature_names = data.columns.tolist()  # Extract feature names from CSV
        data_array = data.values  # Convert to NumPy array for sampling
    elif file_type == 'npy':
        print("Reading data from NumPy file...")
        data_array = np.load(input_file)
        if feature_names is None:
            raise ValueError("Feature names must be provided when using a NumPy file.")
    elif file_type == 'las':
        print("Reading data from LAS file...")
        data_array, feature_names = read_las_file_to_numpy(input_file)
    else:
        raise ValueError("Unsupported file type. Please specify 'csv', 'npy', or 'las'.")

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


def read_csv_file_to_numpy(file_path, features_to_extract):
    """
    Reads a CSV file and extracts the specified features along with coordinates.

    Args:
    - file_path (str): Path to the CSV file.
    - features_to_extract (list of str): List of feature names to extract.

    Returns:
    - np.ndarray: Numpy array containing the extracted features and coordinates.
    """
    df = pd.read_csv(file_path)
    
    # Extract coordinates
    coords = df[['x', 'y', 'z']].values
    
    # Extract the desired features based on names
    feature_data = df[features_to_extract].values
    feature_names = features_to_extract
    
    # Combine coordinates and selected features into a single array
    combined_data = np.hstack((coords, feature_data))
    
    return combined_data, feature_names

def extract_num_classes(file_path, pre_process_data, preprocessed_data_dir=None):
    """
    Extracts the number of unique classes from raw data (LAS, CSV, or NPY) or from preprocessed grid filenames.

    Args:
    - file_path (str): Path to the input LAS, CSV, or NPY file.
    - pre_process_data (bool): If True, extract classes from raw data; if False, extract classes from preprocessed files.
    - preprocessed_data_dir (str, optional): Path to the preprocessed data directory (required if pre_process_data=False).

    Returns:
    - int: The number of unique classes.
    """

    if pre_process_data:
        # Check file extension to handle different file types
        if file_path.lower().endswith('.las'):
            # Load LAS file
            with laspy.read(file_path) as las_data:
                # Convert LAS point data to a NumPy array, assuming the class labels are stored in the last column
                data_array, _  = read_las_file_to_numpy(file_path)
                class_labels = data_array[:, -1]  # Assuming the class label is in the last column
            
        elif file_path.lower().endswith('.csv'):
            # Load CSV file
            data = pd.read_csv(file_path)
            # Assuming the class label is in the last column
            class_labels = data.iloc[:, -1].values

        elif file_path.lower().endswith('.npy'):
            data = np.load(file_path)
            class_labels = data[:, -1]
            
        else:
            raise ValueError("Unsupported file format. Only .las,  .csv or .npy files are supported.")

        # Extract the unique number of classes
        num_classes = len(np.unique(class_labels))
        
    else: 
        # Helper function to extract the common identifier (e.g., 'grid_4998') from the filename
        def get_common_identifier(filename):
            return '_'.join(filename.split('_')[:2])  # Extracts 'grid_4998' from 'grid_4998_small_class_0.npy'

        # Collect filenames and extract identifiers for each scale
        small_files = {get_common_identifier(f): f for f in os.listdir(os.path.join(preprocessed_data_dir, 'small'))}
        medium_files = {get_common_identifier(f): f for f in os.listdir(os.path.join(preprocessed_data_dir, 'medium'))}
        large_files = {get_common_identifier(f): f for f in os.listdir(os.path.join(preprocessed_data_dir, 'large'))}

        # Find common identifiers across all three scales
        common_identifiers = set(small_files.keys()).intersection(medium_files.keys(), large_files.keys())

        if not common_identifiers:
            raise FileNotFoundError("No common grid files found across small, medium, and large scales.")

        # Sort the common identifiers to ensure consistent ordering
        common_identifiers = sorted(common_identifiers)

        # Extract class labels from filenames in the 'small' directory using the common identifiers
        class_labels = []
        for identifier in common_identifiers:
            file_name = f"grid_{identifier}_small_class_0.npy"  # Construct the filename format
            class_label = int(file_name.split('_')[-1].split('.')[0].replace('class_', ''))
            class_labels.append(class_label)
        
        num_classes = len(np.unique(class_labels))

        print(f"Number of common grids: {len(common_identifiers)}")
    
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

"""def clean_point_cloud_data(data_array):
    # Remove rows where any element is NaN or Inf
    valid_mask = ~np.isnan(data_array).any(axis=1) & ~np.isinf(data_array).any(axis=1)
    return data_array[valid_mask]"""



