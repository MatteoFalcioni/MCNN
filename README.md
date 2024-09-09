# Multi-Scale Convolutional Neural Network (MCNN) Training Project

This project contains the implementation of a Multi-Scale Convolutional Neural Network (MCNN) for processing point cloud data and generating corresponding images, based on the methodologies described in the article "Segmentation and Multi-Scale Convolutional Neural
Network-Based Classification of Airborne Laser
Scanner Data" (at https://www.mdpi.com/1424-8220/18/10/3347).

## Table of Contents
- Installation
- Usage
- Structure

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1. **Clone this repository**:

   ```bash
   git clone https://github.com/MatteoFalcioni/MCNN.git
   ```

2. **Install the required packages**:

    ```bash
   pip install -r requirements.txt
    ```


## Usage

To train the MCNN model with default settings:

    python main.py

To train the MCNN model with custom setting you can use the argument parser specify relevant command line arguments. The possible commands are:

- `--batch size`  specifies batch size for training
- `--epochs` specifies the number of training epochs
- `--patience` specifies the number of epochs to wait for an improvement in validation loss before early stopping
- `--learning_rate` specifies the learning rate for the optimizer
- `--learning_rate_decay_epochs` specifies the epochs interval to decay learning rate 
- `--learning_rate_decay_gamma` specifies the learning rate decay factor
- `--save_dir` specifies the directory to save trained models

## Structure

- `MCNN/`
  - `data/`
    - `data_utils.py`           Data processing and loading utilities
  - `models/`
    - `mcnn.py`                Model definitions for MCNN
    - `scnn.py`                 Model definitions for SCNN
  - `scripts/`
    - `train.py`                Training and validation functions
    - `data_preprocessing.py`   Functions for data preprocessing
    - `model_initialization.py` Functions for model initialization
  - `utils/`
    - `config_loader.py`       Configuration management utilities
    - `device_selector.py`      Device selection utilities
    - `plot_utils.py`           Plotting utilities
  - `tests/`
    - `test_training.py`        Unit tests for verifying model training and data loading
    - `test_data_loader.py`     Unit tests for data loader functionality
    - `test_models.py`          Unit tests for model functions
  - `main.py`                   Main script for training the model
  - `requirements.txt`          List of Python dependencies required to run the project
  - `README.md`                 This file, providing an overview and instructions





