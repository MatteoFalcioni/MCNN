import torch
import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.data_utils import prepare_dataloader
from scripts.train import train_epochs
from utils.config_handler import parse_arguments
from utils.utils import select_device


def main():
    # Parse arguments with defaults from config.yaml
    args = parse_arguments()

    # Set device (GPU if available)
    device = select_device()
    print(f"Using device: {device}")

    # Initialize model (always MCNN)
    print("Initializing MultiScaleCNN (MCNN) model...")
    model = MultiScaleCNN().to(device)

    # Prepare DataLoader
    # Replace with actual sets when data will be available
    print("Preparing data loaders...")
    train_loader = prepare_dataloader(args.batch_size)
    val_loader = prepare_dataloader(args.batch_size)

    # Set up CrossEntropy loss function
    criterion = nn.CrossEntropyLoss()

    # Set up optimizer (Stochastic Gradient Descent)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.learning_rate_decay_epochs, gamma=args.learning_rate_decay_gamma)

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
        args.save_dir
    )

    print("train ended")


if __name__ == "__main__":
    main()

