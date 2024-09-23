import argparse
import yaml


def load_config(file_path='config.yaml'):
    """
    Load configuration from a YAML file.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_arguments():
    """
    Sets up the argument parser for training configurations and parses command-line arguments.

    Returns:
    - argparse.Namespace: Parsed arguments as a namespace object.
    """

    # Load configuration defaults from the config file
    config = load_config()

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train a deep learning model.")

    # Adding arguments for hyperparameters and configurations with defaults from the config file
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 16), help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=config.get('epochs', 20),
                        help='Number of epochs to wait for an improvement in validation loss before early stopping')
    parser.add_argument('--patience', type=int, default=config.get('patience', 3),
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=config.get('learning_rate', 0.01),
                        help='Learning rate for the optimizer')
    parser.add_argument('--learning_rate_decay_epochs', type=int, default=config.get('learning_rate_decay_epochs', 1),
                        help='Epochs interval to decay learning rate')
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        default=config.get('learning_rate_decay_factor', 0.0005),
                        help='Learning rate decay factor')
    parser.add_argument('--momentum', type=float, default=config.get('momentum', 0.9),
                        help='Momentum factor for the optimizer')
    parser.add_argument('--models_save_dir', type=str, default=config.get('save_dir', 'models/saved/'),
                        help='Directory to save trained models')
    parser.add_argument('--preprocess_data', action='store_true', default=config.get('preprocess_data', False),
                        help='If set, preprocess raw data for training; otherwise use existing pre-processed data.')
    parser.add_argument('--save_model', action='store_true', default=config.get('save_model', False),
                        help='If set, save the trained model.')
    parser.add_argument('--preprocessed_data_dir', type=str, default=config.get('preprocessed_data_dir', 'data/pre_processed_data'),
                        help='directory where pre-processed data for training is stored.')
    parser.add_argument('--windows_sizes', type=list,
                        default=config.get('windows_sizes', [('small', 2.5), ('medium', 5.0), ('large', 10.0)]),
                        help='directory where pre-processed data for training is stored.')
    parser.add_argument('--load_model', action='store_true', default=config.get('load_model', False),
                        help='If set, loads model from load_model_filepath to perform inference')
    parser.add_argument('--load_model_filepath', type=str,
                        default=config.get('load_model_filepath', 'models/saved/mcnn_model_20240922_231624.pth'),
                        help='directory where pre-processed data for training is stored.')

    # Parsing arguments
    args = parser.parse_args()

    return args


