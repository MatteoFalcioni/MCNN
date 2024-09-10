from utils.point_cloud_data_utils import load_las_data, load_asc_data, read_las_file_to_numpy, numpy_to_dataframe
import os
import pandas as pd
import unittest
import numpy as np


class TestPointCloudDataProcessing(unittest.TestCase):

    def setUp(self):
        self.las_directory = 'data/raw'
        self.asc_file_path = 'data/raw/dtm.asc'
        self.features_to_extract = ['x', 'y', 'z',  'intensity', 'return_number', 'number_of_returns', 'red', 'green', 'blue', 'nir',
                                    'ndvi', 'ndwi', 'ssi', 'l1_b', 'l2_b', 'l3_b', 'planarity_b', 'sphericity_b',
                                    'linearity_b', 'entropy_b', 'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b',
                                    'N_h', 'delta_z_fl']
        self.numpy_array = np.array([
            [681999.979, 4931411.238, 42.146, 182, 0.18326766788959503, -0.14921404421329498],
            [681999.999, 4931410.997, 42.203, 202, 0.21067164838314056, -0.1748277097940445],
            [681999.955, 4931409.817, 42.217, 1122, 0.1847124844789505, -0.14715950191020966]
        ])
        self.column_names = ['x', 'y', 'z', 'intensity', 'feature1', 'feature2']

    def test_load_las_data(self):
        # Test loading LAS data
        las_files = [f for f in os.listdir(self.las_directory) if f.endswith('.las')]
        if las_files:
            points = load_las_data(os.path.join(self.las_directory, las_files[0]))
            self.assertIsInstance(points, np.ndarray, "Loaded LAS data is not a numpy array.")
            self.assertEqual(points.shape[1], 3, "LAS data does not have three coordinate columns.")
        else:
            self.skipTest("No LAS files found for testing `load_las_data`.")

    def test_load_asc_data(self):
        # Test loading ASC data
        try:
            dtm_data = load_asc_data(self.asc_file_path)
            self.assertIsInstance(dtm_data, np.ndarray, "Loaded ASC data is not a numpy array.")
            self.assertGreater(dtm_data.shape[0], 0, "Loaded ASC data is unexpectedly empty.")
        except FileNotFoundError:
            self.skipTest("ASC file not found for testing `load_asc_data`.")

    def test_read_las_file_to_numpy(self):
        # Test the read_las_file_to_numpy function
        sample_file = os.path.join(self.las_directory, 'features_F.las')  # Use a sample file
        numpy_array = read_las_file_to_numpy(sample_file, features_to_extract=self.features_to_extract)

        # Ensure the result is a NumPy array
        self.assertIsInstance(numpy_array, np.ndarray, "Result is not a NumPy array.")

        # Ensure the shape of the array is as expected (at least 4 columns: x, y, z, features)
        self.assertGreaterEqual(numpy_array.shape[1], 4, "NumPy array does not have the expected number of columns.")

        # Check some values to ensure data was loaded correctly (sample values for the first few points)
        print("Sample data from NumPy array (first 5 rows):")
        print(numpy_array[:5])

    def test_numpy_to_dataframe(self):
        # Convert numpy array to DataFrame
        df = numpy_to_dataframe(self.numpy_array, self.column_names)

        # Check the result is a DataFrame
        self.assertIsInstance(df, pd.DataFrame, "The result is not a pandas DataFrame.")

        # Check the DataFrame has the correct number of rows and columns
        self.assertEqual(df.shape[0], self.numpy_array.shape[0], "The number of rows in the DataFrame does not match the input NumPy array.")
        self.assertEqual(df.shape[1], self.numpy_array.shape[1], "The number of columns in the DataFrame does not match the input NumPy array.")

        # Check the DataFrame columns have the expected names
        self.assertListEqual(list(df.columns), self.column_names, "The DataFrame columns do not match the expected names.")

        # Check if the values are correctly copied from NumPy array
        np.testing.assert_array_almost_equal(df.values, self.numpy_array, decimal=6, err_msg="The DataFrame values do not match the original NumPy array values.")

        # Print sample data for manual verification
        print("Sample data from DataFrame (first 5 rows):")
        print(df.head())

