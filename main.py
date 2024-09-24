import torch
import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.train_data_utils import prepare_dataloader
from scripts.train import train_epochs
from scripts.inference import inference
from utils.config_handler import parse_arguments
from utils.point_cloud_data_utils import read_las_file_to_numpy, sample_data
import numpy as np


def main():
    # Parse arguments with defaults from config.yaml
    args = parse_arguments()

    # Set device (GPU if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model (always MCNN)
    print("Initializing MultiScaleCNN (MCNN) model...")
    model = MultiScaleCNN(channels=10, classes=5).to(device)  # Make sure to set classes correctly

    # Prepare DataLoader
    print("Preparing data loaders...")

    labeled_filepath = 'data/combined_data/sampled/sampled_data20000.npy'

    train_loader, val_loader = prepare_dataloader(
        batch_size=args.batch_size,
        data_dir=labeled_filepath,
        grid_save_dir='data/pre_processed_data_new',
        pre_process_data=True,
        window_sizes=args.window_sizes,
        grid_resolution=128,
        channels=7,
        save_grids=True,
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
    print("Starting training process with early stopping...")
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
        save_dir="models/saved/"
    )

    print("Training finished")

"""    # Run inference on a sample
    print("Starting inference process...")
    data_array, _ = read_las_file_to_numpy(labeled_filepath)
    # need to remap labels to match the ones in training. Maybe consider remapping already when doing las -> numpy ?
    remapped_array, _ = remap_labels(data_array)
    sample_array = remapped_array[np.random.choice(remapped_array.shape[0], 200, replace=False)]
    ground_truth_labels = sample_array[:, -1]  # Assuming labels are in the last column

    load_model = args.load_model   # load model for inference or train a new one?
    model_path = args.load_model_filepath
    if load_model:
        loaded_model = MultiScaleCNN(channels=10, classes=5).to(device)
        # Load the saved model state dictionary
        loaded_model.load_state_dict(torch.load(model_path, map_location=device))
        model = loaded_model
        # Set model to evaluation mode (important for inference)
        model.eval()

    predicted_labels, precision = inference(
        model,
        data_array=sample_array,
        window_sizes=[2.5, 5, 10],
        grid_resolution=128,
        channels=10,
        device=device,
        true_labels=ground_truth_labels
    )

    print(f"Predicted Labels: {predicted_labels}")"""


if __name__ == "__main__":
    main()


# important: sometimes you use window sizes as list like window_sizes=[2.5, 5, 10], sometimes as tuples like
# window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]). Most importantly, in multiscale_grids and
# in the dataloader it's  a tuple, while in inference or training it's not. Use same standard for everything.

# Also, we might want to change the channels parameter to a list of features chosen by the user. that way we can
# get channels from the lenght of features_to_train (for example) and avoid hard coding it everytime.

