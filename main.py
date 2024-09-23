import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.train_data_utils import prepare_dataloader
from scripts.train import train_epochs
from scripts.inference import inference
from utils.config_handler import parse_arguments
from utils.device_utils import select_device
from utils.point_cloud_data_utils import read_las_file_to_numpy
import numpy as np


def main():
    # Parse arguments with defaults from config.yaml
    args = parse_arguments()

    # Set device (GPU if available)
    device = select_device()
    print(f"Using device: {device}")

    # Initialize model (always MCNN)
    print("Initializing MultiScaleCNN (MCNN) model...")
    model = MultiScaleCNN(channels=10, classes=5).to(device)  # Make sure to set classes correctly

    # Prepare DataLoader
    print("Preparing data loaders...")
    labeled_filepath = 'data/raw/labeled_FSL.las'
    train_loader, val_loader = prepare_dataloader(
        batch_size=args.batch_size,
        data_dir=labeled_filepath,
        pre_process_data=False,
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

    # Run inference on a sample
    print("Starting inference process...")
    data_array, _ = read_las_file_to_numpy(labeled_filepath)
    sample_array = data_array[np.random.choice(data_array.shape[0], 200, replace=False)]

    predicted_labels = inference(
        model,
        data_array=sample_array,
        window_sizes=[2.5, 5, 10],
        grid_resolution=128,
        channels=10,
        device=device
    )

    print(f"Predicted Labels: {predicted_labels}")


if __name__ == "__main__":
    main()

