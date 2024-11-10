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
import torch.multiprocessing as mp
import sys


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
            small_grids, medium_grids, large_grids, labels, _ = batch
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


def inference_without_ground_truth(model, dataloader, device, data_file, model_path, save_subfolder="predictions"):
    """
    Runs inference and writes predictions directly to a LAS file in a batch-wise manner.

    Args:
    - model (nn.Module): The trained PyTorch model.
    - dataloader (DataLoader): DataLoader containing the point cloud data for inference.
    - device (torch.device): Device to perform inference on (CPU or GPU).
    - data_file (str): Path to the input file (used for naming the output file).
    - model_path (str): Path to the model to be used for inference.
    - save_subfolder (str): Subdirectory of the model's folder where the LAS file with predictions will be saved.

    Returns:
    - las_file_path (str): File path to the saved LAS file. 
    """
    
    mp.set_sharing_strategy('file_system')  # trying to fix too many open files error

    model.eval()
    # Set up output path
    model_save_folder = os.path.dirname(model_path)  # Get the directory in which the model file is stored (the parent directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_file_name = f"{os.path.splitext(os.path.basename(data_file))[0]}_CNN_{timestamp}.las"
    
    # Create timestamped folder inside the predictions subfolder
    save_dir = os.path.join(model_save_folder, save_subfolder)
    os.makedirs(save_dir, exist_ok=True)
    
    las_file_path = os.path.join(save_dir, pred_file_name)

    # Open the input LAS file to copy header information
    original_file = laspy.read(data_file)
    header = original_file.header
    
    # Check and add 'label' as needed
    if 'label' not in header.point_format.dimension_names:
        extra_dims = [laspy.ExtraBytesParams(name="label", type=np.int8)]
        header.add_extra_dims(extra_dims)

    # Initialize the label field with -1 values (-1 = not classified)
    total_points = len(original_file.x)
    label_array = np.full(total_points, -1, dtype=np.int8)
    
    all_predictions = []  # List to store predictions
    all_indices = []  # List to store indices

    # Perform inference 
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Performing inference", ascii=True, dynamic_ncols=True, file=sys.stdout):
            if batch is None:
                continue

            small_grids, medium_grids, large_grids, _, indices = batch
            small_grids, medium_grids, large_grids = (
                small_grids.to(device), medium_grids.to(device), large_grids.to(device)
            )

            # Run model inference
            outputs = model(small_grids, medium_grids, large_grids)
            preds = torch.argmax(outputs, dim=1)

            # Store predictions and indices
            all_predictions.append(preds.cpu().numpy())
            all_indices.append(indices)
    
    # Concatenate all predictions and indices
    all_predictions = np.concatenate(all_predictions)
    all_indices = np.concatenate(all_indices) 
    
    # Directly assign predictions to the label array
    label_array[all_indices] = all_predictions
            
    # Create a new LasData object and assign fields for all point data and the label array
    new_las = laspy.LasData(header)
    new_las.points = original_file.points  # Copy original points
    new_las.label = label_array            # Assign the labels to the new dimension
    
    new_las.write(las_file_path)    # Write to the new file 
        
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