from utils.point_cloud_data_utils import sample_data

sample_data(input_file='data/raw/labeled_FSL.las', file_type='las', sample_size=5000, save_dir='data/sampled/', save=True)