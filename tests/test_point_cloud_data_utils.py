from utils.point_cloud_data_utils import load_las_data, load_asc_data, read_feature_las_files
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
        self.n_samples = 50  # Use a smaller number of samples for efficient testing
        self.csv_output_dir = 'test/test_csv_files'

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

    def test_read_feature_las_files(self):
        # Sample the DataFrame to speed up testing
        dataframes = read_feature_las_files(
            las_directory=self.las_directory,
            features_to_extract=self.features_to_extract,
            save_to_csv=True,
            sample_size=self.n_samples,
            csv_output_dir=self.csv_output_dir
        )

        # Check that some DataFrames are returned
        self.assertGreater(len(dataframes), 0, "No DataFrames returned from read_feature_las_files.")

        for df in dataframes:
            # Ensure the result is a DataFrame
            self.assertIsInstance(df, pd.DataFrame, "Result is not a pandas DataFrame.")

            # Ensure 'x', 'y', 'z' are present
            self.assertIn('x', df.columns, "'x' column missing in DataFrame.")
            self.assertIn('y', df.columns, "'y' column missing in DataFrame.")
            self.assertIn('z', df.columns, "'z' column missing in DataFrame.")

            # Ensure all specified features are included or reported as missing
            missing_features = []
            for feature in self.features_to_extract:
                if feature not in ['x', 'y', 'z']:
                    if feature not in df.columns:
                        missing_features.append(feature)

            # Check if all missing features were correctly reported
            print(f"Missing features (if any): {missing_features}")

            # Ensure the DataFrame is not empty
            self.assertFalse(df.empty, "DataFrame is unexpectedly empty.")

        # Check if CSV files are saved correctly
        for las_file in os.listdir(self.las_directory):
            if las_file.endswith('_F.las'):
                csv_filename = os.path.splitext(las_file)[0] + '.csv.gz'
                csv_path = os.path.join(self.csv_output_dir, csv_filename)
                self.assertTrue(os.path.exists(csv_path), f"CSV file {csv_path} was not saved correctly.")

                # Load the CSV and check its contents
                loaded_df = pd.read_csv(csv_path, compression='gzip')
                self.assertIsInstance(loaded_df, pd.DataFrame, "Loaded CSV is not a pandas DataFrame.")

                # Sample the loaded DataFrame as well for consistency in testing
                sampled_loaded_df = loaded_df.sample(n=self.n_samples, random_state=42)

                # Ensure that the columns match
                self.assertEqual(list(sampled_loaded_df.columns), list(df.columns),
                                 "CSV columns do not match the original DataFrame.")

