from utils.point_cloud_data_utils import sample_data, combine_and_save_csv_files, read_file_to_numpy
import pandas as pd

sample_data(input_file='data/training_data/train_21.csv', sample_size=2000000, save_dir='data/sampled/', save=True)

data_array, features = read_file_to_numpy('data/training_data/to_compare/32_684000_4930500_FP21.las')
print(f'features {features}')

'''csv_file_path = 'data/sampled/sampled_data_1000000.csv'  
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
print(nan_per_column)'''