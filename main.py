import torch
import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.train_data_utils import prepare_dataloader, initialize_weights
from scripts.train import train_epochs
from scripts.inference import inference
from utils.config_handler import parse_arguments
from utils.point_cloud_data_utils import read_las_file_to_numpy, remap_labels, extract_num_classes, extract_num_channels
import numpy as np


def main():
    # Parse arguments with defaults from config.yaml
    args = parse_arguments()
    print(f'window sizes: {args.window_sizes}')

    # Set device (GPU if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Extract user-selected features from the config, only if pre-processing data: 
    if args.preprocess_data:
        features_to_use = args.features_to_use  # List of features chosen by the user
        num_channels = len(features_to_use)  # Determine the number of channels based on selected features
    else:
        features_to_use = None    # no need to select features if training data already exists
        num_channels=extract_num_channels(args.preprocessed_data_dir)   # extract number of channels from existing grids

    # determine the number of classes
    num_classes = extract_num_classes(raw_file_path=args.raw_data_filepath, pre_process_data=args.preprocess_data, preprocessed_data_dir=args.preprocessed_data_dir)   

    # Initialize model 
    print("Initializing MultiScaleCNN (MCNN) model...")
    model = MultiScaleCNN(channels=num_channels, classes=num_classes).to(device)  
    model.apply(initialize_weights)     # initialize model weights (optional, but recommended)

    # Prepare DataLoader
    print("Preparing data loaders...")

    train_loader, val_loader = prepare_dataloader(
        batch_size=args.batch_size,
        data_dir=args.raw_data_filepath,
        grid_save_dir=args.preprocessed_data_dir,
        pre_process_data=args.preprocess_data,
        window_sizes=args.window_sizes,
        grid_resolution=128,
        features_to_use=features_to_use,
        train_split=0.8
    )

    # Set up CrossEntropy loss function
    criterion = nn.CrossEntropyLoss()

    # Set up optimizer (Stochastic Gradient Descent)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.learning_rate_decay_epochs,
                                          gamma=args.learning_rate_decay_factor)

    # Start training with early stopping
    print("Starting training process...")
    train_epochs(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        args.epochs,
        args.patience,
        device,
        save=True,
        plot_dir='results/plots/',
        model_save_dir="models/saved/"
    )

    print("Training finished")

    # Run inference on a sample
    print("Starting inference process...")

    data_array, _ = read_las_file_to_numpy(args.raw_data_filepath)
    # need to remap labels to match the ones in training. Maybe consider remapping already when doing las -> numpy ?
    remapped_array, _ = remap_labels(data_array)
    sample_array = remapped_array[np.random.choice(remapped_array.shape[0], 200, replace=False)]
    ground_truth_labels = sample_array[:, -1]  # Assuming labels are in the last column

    load_model = args.load_model   # whether to load model for inference or train a new one
    model_path = args.load_model_filepath
    if load_model:
        loaded_model = MultiScaleCNN(channels=num_channels, classes=num_classes).to(device)
        # Load the saved model state dictionary
        loaded_model.load_state_dict(torch.load(model_path, map_location=device))
        model = loaded_model
        # Set model to evaluation mode (important for inference)
        model.eval()

    predicted_labels = inference(
        model,
        data_array=sample_array,
        window_sizes=[('small', 2.5), ('medium', 5.0), ('large', 10.0)],
        grid_resolution=128,
        channels=num_channels,
        device=device,
        true_labels=ground_truth_labels, 
        save_file=f'results/labels/'
    )

    for i in len(ground_truth_labels):
        print(f'predicted label: {predicted_labels[i]}, true label: {ground_truth_labels[i]}')
    print('process ended successfully.')



if __name__ == "__main__":
    main()



