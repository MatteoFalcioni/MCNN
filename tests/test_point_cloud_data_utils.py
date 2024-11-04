from utils.point_cloud_data_utils import load_las_data, load_asc_data, read_las_file_to_numpy, numpy_to_dataframe
from utils.point_cloud_data_utils import combine_and_save_csv_files, subtiler
import os
import pandas as pd
import unittest
import numpy as np
import laspy


'''class TestPointCloudDataUtils(unittest.TestCase):

    def setUp(self):
        self.las_directory = 'data/raw'
        self.asc_file_path = 'data/raw/dtm.asc'
        self.features_to_extract = ['x', 'y', 'z',  'intensity', 'return_number', 'number_of_returns', 'red', 'green', 'blue', 'nir',
                                    'ndvi', 'ndwi', 'ssi', 'l1_b', 'l2_b', 'l3_b', 'planarity_b', 'sphericity_b',
                                    'linearity_b', 'entropy_b', 'theta_b', 'theta_variance_b', 'mad_b', 'delta_z_b',
                                    'N_h', 'delta_z_fl']
        self.csv_files = ['data/classes/buildings.csv', 'data/classes/cars.csv', 'data/classes/grass.csv',
                          'data/classes/rail.csv', 'data/classes/roads.csv', 'data/classes/trees.csv']
        self.csv_save_dir = 'data/combined_data'  # Directory to save combined data

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
        sample_file = os.path.join(self.las_directory, 'features_F.las')
        numpy_array, feature_names = read_las_file_to_numpy(sample_file, features_to_extract=self.features_to_extract)

        # Ensure the result is a NumPy array
        self.assertIsInstance(numpy_array, np.ndarray, "Result is not a NumPy array.")

        # Ensure the shape of the array is as expected (at least 4 columns: x, y, z, features)
        self.assertGreaterEqual(numpy_array.shape[1], 4, "NumPy array does not have the expected number of columns.")
        self.assertGreater(len(feature_names), 0, "No feature names returned.")

        # Check that the first three feature names are 'x', 'y', 'z'
        self.assertEqual(feature_names[:3], ['x', 'y', 'z'], "The first three features should be 'x', 'y', 'z'.")

        # Ensure that all specified features to extract are either in the list or not available
        for feature in self.features_to_extract:
            if feature in ['x', 'y', 'z']:
                continue
            self.assertIn(feature, feature_names, f"Feature '{feature}' is not found in the returned feature names.")

        # Check some values to ensure data was loaded correctly (sample values for the first few points)
        print("Sample data from NumPy array (first 5 rows):")
        print(numpy_array[:5])
        print("Feature names extracted:")
        print(feature_names)

        # check that segment_id and label are correctly extracted from a labeled sample file
        labeled_file = os.path.join(self.las_directory, 'labeled_FSL.las')
        lbl_array, feature_names_lbl = read_las_file_to_numpy(labeled_file, features_to_extract=['x', 'y', 'z', 'segment_id', 'label'])
        self.assertIn('segment_id', feature_names_lbl, "'segment_id' is not found in the returned feature names.")
        self.assertIn('label', feature_names_lbl, "'label' is not found in the returned feature names.")

    def test_numpy_to_dataframe(self):
        sample_file = os.path.join(self.las_directory, 'features_F.las')
        numpy_array, feature_names = read_las_file_to_numpy(sample_file, features_to_extract=self.features_to_extract)

        # Convert numpy array to DataFrame
        df = numpy_to_dataframe(numpy_array, feature_names=self.features_to_extract)

        # Check the result is a DataFrame
        self.assertIsInstance(df, pd.DataFrame, "The result is not a pandas DataFrame.")

        # Check the DataFrame has the correct number of rows and columns
        self.assertEqual(df.shape[0], numpy_array.shape[0], "The number of rows in the DataFrame does not match the input NumPy array.")
        self.assertEqual(df.shape[1], numpy_array.shape[1], "The number of columns in the DataFrame does not match the input NumPy array.")

        # Check the DataFrame columns have the expected names
        self.assertListEqual(list(df.columns), feature_names, "The DataFrame columns do not match the expected names.")

        # Check if the values are correctly copied from NumPy array
        np.testing.assert_array_almost_equal(df.values, numpy_array, decimal=6, err_msg="The DataFrame values do not match the original NumPy array values.")

        # Print sample data for manual verification
        print("Sample data from DataFrame (first 5 rows):")
        print(df.head())

    def test_combine_and_save_csv_files(self):
        """ Test combining and saving CSV files into a single NumPy array. """

        # Test combining without saving
        combined_data = combine_and_save_csv_files(self.csv_files, save=False)

        # Check that the result is a NumPy array
        self.assertIsInstance(combined_data, np.ndarray, "The result is not a NumPy array.")

        # Ensure that the array has some rows
        self.assertGreater(combined_data.shape[0], 0, "Combined array is unexpectedly empty.")

        # Test combining with saving
        combined_data_saved = combine_and_save_csv_files(self.csv_files, save=True, save_dir=self.csv_save_dir)

        # Ensure the combined data was saved
        saved_file_path = os.path.join(self.csv_save_dir, 'combined_data.npy')
        self.assertTrue(os.path.exists(saved_file_path), "Combined data was not saved correctly.")

        # Load the saved file and compare it with the original combined data
        loaded_data = np.load(saved_file_path)
        np.testing.assert_array_almost_equal(combined_data_saved, loaded_data, decimal=6,
                                             err_msg="Loaded data does not match the originally combined data.")
'''

class TestSubtiler(unittest.TestCase):

    def setUp(self):
        # Set up test directory and parameters
        self.test_dir = 'tests/test_subtiler'
        self.output_dir = os.path.join(self.test_dir, '32_683500_4924500_FP21_050')
        self.tile_size = 50
        os.makedirs(self.test_dir, exist_ok=True)

    def test_subtiler(self):
        # Print the features in the original LAS file for reference
        original_file = os.path.join(self.test_dir, '32_683500_4924500_FP21.las')
        las = laspy.read(original_file)
        print("\nFeatures (dimensions) in the original LAS file:")
        for dimension_name in las.point_format.dimension_names:
            print(f"- {dimension_name}")
        
        # Run the subtiler function on the test directory
        subtiler(directory=self.test_dir, tile_size=self.tile_size)

        # Check if the output directory is created
        self.assertTrue(os.path.exists(self.output_dir), "Output directory was not created")

        las_files = [f for f in os.listdir(self.output_dir) if f.endswith('.las')]

        file_idx = 0
        # Check that each subtile retains features and labels
        for las_file in las_files:
            las = laspy.read(os.path.join(self.output_dir, las_file))
            self.assertTrue(hasattr(las, 'points'), "Subtile missing point data")
            self.assertGreater(len(las.points), 0, "Subtile has no points")
            self.assertTrue(hasattr(las.points, 'intensity'), "Subtile missing intensity feature")
            if file_idx < 5:
                print("\nFeatures (dimensions) in the subtile LAS file:")
                for dimension_name in las.point_format.dimension_names:
                    print(f"- {dimension_name}")
            file_idx += 1
    
    def tearDown(self):
    # Clean up generated files and directories after test
        for root, dirs, files in os.walk(self.output_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.output_dir)