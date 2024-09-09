import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def load_las_features(file_path):
    """
    Carica il file LAS con le features calcolate e restituisce un DataFrame con le features.

    Args:
    - file_path (str): Percorso al file LAS con features.

    Returns:
    - pd.DataFrame: DataFrame contenente le features del file LAS.
    """
    las = laspy.read(file_path)

    # Extract the feature columns
    features = {
                'x': las.x,
                'y': las.y,
                'z': las.z,
                'intensity': las.intensity,
                'ndvi': las.ndvi,
                'ndwi': las.ndwi,
                'ssi': las.ssi,
                'l1_a': las.l1_a,
                'l2_a': las.l2_a,
                'l3_a': las.l3_a,
                'planarity_a': las.planarity_a,
                'sphericity_a': las.sphericity_a,
                'linearity_a': las.linearity_a,
                'entropy_a': las.entropy_a,
                'theta_a': las.theta_a,
                'theta_variance_a': las.theta_variance_a,
                'mad_a': las.mad_a,
                'delta_z_a': las.delta_z_a,
                'l1_b': las.l1_b,
                'l2_b': las.l2_b,
                'l3_b': las.l3_b,
                'planarity_b': las.planarity_b,
                'sphericity_b': las.sphericity_b,
                'linearity_b': las.linearity_b,
                'entropy_b': las.entropy_b,
                'theta_b': las.theta_b,
                'theta_variance_b': las.theta_variance_b,
                'mad_b': las.mad_b,
                'delta_z_b': las.delta_z_b,
                'N_h': las.N_h,
                'delta_z_fl': las.delta_z_fl
    }

    # Convert to a DataFrame for easier manipulation
    features_df = pd.DataFrame(features)

    return features_df


def convert_dataframe_to_numpy(features_df):
    """
    Converts the feature DataFrame into a NumPy array.

    Args:
    - features_df (pd.DataFrame): DataFrame containing all points and their features.

    Returns:
    - data_array (numpy.ndarray): A NumPy array where each row represents a point,
                                  and columns represent x, y, z, and all other features.
    """
    # Convert entire DataFrame to a NumPy array
    data_array = features_df.to_numpy()
    return data_array


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

