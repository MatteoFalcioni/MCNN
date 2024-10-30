import torch
import numpy as np
import pandas as pd
import laspy
from laspy.file import File
from datetime import datetime
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


def inference(model, dataloader, device, class_names, model_save_folder, inference_file_path, save=False):
    """
    Runs inference on the provided data, returns the confusion matrix and classification report,
    and optionally saves them to the 'inference' subfolder of the model save directory.

    Args:
    - model (nn.Module): The trained PyTorch model.
    - dataloader (DataLoader): DataLoader containing the data for inference.
    - device (torch.device): The device (CPU or GPU) where computations will be performed.
    - class_names (list): List of class names for displaying in the confusion matrix.
    - model_save_folder (str): The folder where the model is saved and where inference results will be saved.
    - inference_file_path (str): The path to the file used for inference.
    - save (bool): If True, saves the confusion matrix and classification report.

    Returns:
    - conf_matrix (np.ndarray): The confusion matrix.
    - class_report (str): The classification report as a string.
    """
    # Check that model_save_folder is the actual directory and not a filepath to the model
    if model_save_folder.endswith('.pth'):
        model_save_folder = os.path.dirname(model_save_folder)
    
    # Create 'inference' subfolder within the model save folder, with a time stamp suffix to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_dir = os.path.join(model_save_folder, f'inference_{timestamp}')
    os.makedirs(inference_dir, exist_ok=True)

    model.eval()  # Set model to evaluation mode
    all_preds = []  # To store all predictions
    all_labels = []  # To store all true labels

    with torch.no_grad():  # No gradient calculation during inference
        for batch in tqdm(dataloader, desc="Performing inference"):
            if batch is None:
                continue

            # Unpack the batch
            small_grids, medium_grids, large_grids, labels = batch
            small_grids, medium_grids, large_grids, labels = (
                small_grids.to(device), medium_grids.to(device), large_grids.to(device), labels.to(device)
            )

            # Forward pass to get outputs
            outputs = model(small_grids, medium_grids, large_grids)
            preds = torch.argmax(outputs, dim=1)  # Get predicted labels

            # Append predictions and true labels to lists
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate lists to arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    class_report = classification_report(all_labels, all_preds)

    # Save the confusion matrix and classification report together
    if save:
        save_inference_results(conf_matrix, class_report, inference_dir, class_names)
        # Save a log of the file used for inference
        log_file_path = os.path.join(inference_dir, 'inference_log.txt')
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Inference performed on file: {inference_file_path}\n")

    return conf_matrix, class_report

def inference_without_ground_truth(model, dataloader, device, data_file, model_save_folder, save_folder="predictions"):
    """
    Runs inference and writes predictions directly to a LAS file in a batch-wise manner.

    Args:
    - model (nn.Module): The trained PyTorch model.
    - dataloader (DataLoader): DataLoader containing the point cloud data for inference.
    - device (torch.device): Device to perform inference on (CPU or GPU).
    - data_file (str): Path to the input file (used for naming the output file).
    - model_save_folder (str): The folder where the model is saved.
    - save_folder (str): Subdirectory of model_save_folder where the LAS file with predictions will be saved.

    Returns:
    - las_file_path (str): File path to the saved LAS file. 
    """

    model.eval()
    # Set up output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_file_name = f"{os.path.splitext(os.path.basename(data_file))[0]}_pred_{timestamp}.las"
    save_dir = os.path.join(model_save_folder, save_folder)
    os.makedirs(save_dir, exist_ok=True)
    las_file_path = os.path.join(save_dir, pred_file_name)

    # Open the input LAS file to copy header information
    original_file = laspy.read(data_file)
    header = original_file.header
    
    # Check and add classification as needed
    if 'classification' not in header.point_format.dimension_names:
        print('classification not in header.point_format')
        extra_dims = [laspy.ExtraBytesParams(name="classification", type=np.int8)]
        header.add_extra_dims(extra_dims)
        
    # Initialize the LAS file with header and directly set up fields without full memory array
    with laspy.open(las_file_path, mode="w", header=header) as las:
        las.x = original_file.x
        las.y = original_file.y
        las.z = original_file.z
        las.intensity = original_file.intensity
        las.return_number = original_file.return_number
        las.number_of_returns = original_file.number_of_returns

        # Set classification field to -1 directly in the file
        las.classification = np.full(len(original_file.x), -1, dtype=np.int8)
        print("Initial classification set to -1 for all points.")
        print(f'full classification field after init to -1: {las.classification}')
        

        # Perform inference in batches
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Performing inference"):
                if batch is None:
                    continue

                small_grids, medium_grids, large_grids, _, indices = batch
                small_grids, medium_grids, large_grids = (
                    small_grids.to(device), medium_grids.to(device), large_grids.to(device)
                )

                outputs = model(small_grids, medium_grids, large_grids)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                # Debugging statements to track progress
                print(f"Batch indices: {indices}")
                print(f"Predicted labels for batch: {preds}")
                print(f"Classification field before update at indices {indices}: {las.classification[indices]}")

                # Directly assign predictions to the file's classification field
                las.classification[indices] = preds

                # Check classification after update
                print(f"Classification field after update at indices {indices}: {las.classification[indices]}")

    print(f'full classification field: {las.classification}')

    print(f"Predictions saved to {las_file_path}")
    
    # Verify saved content
    saved_file = laspy.read(las_file_path)
    saved_labels = saved_file.classification
    print(f"Number of points saved: {len(saved_labels)}")
    print(f"First 100 classifications: {saved_labels[:100]}")  # Preview
        
    return las_file_path



def save_inference_results(conf_matrix, class_report, save_dir, class_names):
    """
    Saves the confusion matrix as an image and the classification report as a CSV file.

    Args:
    - conf_matrix (np.array): The confusion matrix.
    - class_report (str): The classification report as a string.
    - save_dir (str): Directory where the files should be saved.
    - class_names (list): List of class names for labeling the confusion matrix.
    """
    os.makedirs(save_dir, exist_ok=True)

        # Save confusion matrix as an image using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 7))
    cax = ax.matshow(conf_matrix, cmap='Blues')
    plt.colorbar(cax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Annotate each cell with the count value
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(conf_matrix[i, j], 'd'), ha="center", va="center")

    confusion_matrix_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, bbox_inches='tight')
    print(f"Confusion matrix saved at {confusion_matrix_path}")
    plt.close()

    # Parse and save classification report as CSV
    report_data = []
    lines = class_report.split('\n')
    
    for line in lines[2:-3]:
        row_data = line.split()
        
        # Ensure the line has enough elements (5 in this case)
        if len(row_data) == 5:
            report_data.append({
                'class': row_data[0],
                'precision': row_data[1],
                'recall': row_data[2],
                'f1-score': row_data[3],
                'support': row_data[4],
            })
        else:
            # print(f"Skipping line due to unexpected format: {line}")
            print('')
    
    df = pd.DataFrame.from_records(report_data)
    class_report_path = os.path.join(save_dir, 'classification_report.csv')
    df.to_csv(class_report_path, index=False)
    print(f"Classification report saved at {class_report_path}")