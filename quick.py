from utils.point_cloud_data_utils import sample_data, combine_and_save_csv_files
import pandas as pd

# sample_data(input_file='data/training_data/train_21.csv', sample_size=500000, save_dir='data/sampled/', save=True)

csv_file_path = 'data/sampled/sampled_data_500000.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# Check for NaN values in the entire DataFrame
nan_mask = df.isna()

# Count the total number of NaN values
total_nan = nan_mask.sum().sum()

# Count the number of NaN values per column
nan_per_column = nan_mask.sum()

# Print the results
print(f"Total NaN values in the CSV file: {total_nan}")
print("Number of NaN values per column:")
print(nan_per_column)