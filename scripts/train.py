import torch
from utils.plot_utils import plot_loss
from utils.train_data_utils import save_model
from tqdm import tqdm


def train(model, dataloader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0.0  # This will accumulate the loss for the entire epoch
    running_loss = 0.0  # This will be used for logging every 10 batches
    total_batches = len(dataloader)  # Total number of batches

    # Initialize tqdm progress bar for the training loop
    progress_bar = tqdm(enumerate(dataloader), total=total_batches, desc='Training', leave=False)

    for i, (small_grids, medium_grids, large_grids, labels) in progress_bar:
        small_grids, medium_grids, large_grids, labels = (
            small_grids.to(device), medium_grids.to(device), large_grids.to(device), labels.to(device)
        )

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(small_grids, medium_grids, large_grids)

        # Compute loss
        loss = criterion(outputs, labels)

        # Check for NaN in loss
        # Check for NaN or Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("NaN or Inf loss encountered during training. Stopping training.")

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # Accumulate total loss
        running_loss += loss.item()  # Accumulate running loss for logging

        # Update progress bar description every batch
        if i % 10 == 9:
            progress_bar.set_postfix({'Batch Loss': running_loss / 10})
            running_loss = 0.0  # Reset running loss after logging

    # Return the average loss over the epoch
    return total_loss / total_batches


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set and returns the average loss.

    Args:
    - model (nn.Module): The PyTorch model to be trained.
    - dataloader (DataLoader): DataLoader object containing the validation data.
    - criterion (nn.Module): The loss function to optimize.
    - device (torch.device): The device (CPU or GPU) to perform computations on.

    Returns:
    - float: the average validation loss for each epoch
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for small_grids, medium_grids, large_grids, labels in dataloader:
            small_grids, medium_grids, large_grids, labels = (
                small_grids.to(device), medium_grids.to(device), large_grids.to(device), labels.to(device)
            )

            # Forward pass with three inputs
            outputs = model(small_grids, medium_grids, large_grids)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    return val_loss / len(dataloader)


def train_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, device, model_save_dir='models/saved/', plot_dir='results/plots/', save=False):
    """
    Trains the MCNN model over multiple epochs with early stopping and applies learning rate decay.
    After each epoch, the model is evaluated on a validation set, and training stops if the validation
    loss does not improve for a specified number of epochs (patience). Saves the trained model after
    the training process completes.

    Args:
    - model (nn.Module): The MCNN model to be trained.
    - train_loader (DataLoader): DataLoader object for the training data.
    - val_loader (DataLoader): DataLoader object for the validation data.
    - criterion (nn.Module): The loss function to optimize.
    - optimizer (optim.Optimizer): The optimizer to use for training.
    - scheduler (optim.lr_scheduler): Learning rate scheduler for adjusting the learning rate.
    - epochs (int): Number of epochs to train the model.
    - device (torch.device): The device (CPU or GPU) to perform computations on.
    - save_dir (str): Directory to save the trained model.
    - patience (int): Number of epochs to wait for an improvement in validation loss before early stopping.
    - plot_dir (str): Directory to save the loss plots. Default is 'results/plots/'.
    - save (bool): boolean value to allow or disallow saving of the model after training. Default is False.
    """

    train_losses = []  # To store training losses
    val_losses = []  # To store validation losses

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # Training step
        train_loss = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validation step
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter if improvement
        else:
            patience_counter += 1
            print(f'No improvement in validation loss for {patience_counter} epoch(s).')

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience_counter} epochs with no improvement.")
            break

        # Step the scheduler to adjust the learning rate
        scheduler.step()

    # Save the MCNN model after training
    if save:
        print("saving trained model...")
        save_model(model, model_save_dir)
        print("model saved successfully.")

    # Plot the losses at the end of training
    plot_loss(train_losses, val_losses, plot_dir)



