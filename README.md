# Multi-Scale Convolutional Neural Network (MCNN) Training Project

This project contains the implementation of a Multi-Scale Convolutional Neural Network (MCNN) for processing point cloud data and generating corresponding images, based on the methodologies described in the articles
"Segmentation and Multi-Scale Convolutional Neural Network-Based Classification of Airborne Laser
Scanner Data" (2018) - at https://www.mdpi.com/1424-8220/18/10/3347 - and "A Convolutional Neural Network-Based 3D Semantic
Labeling Method for ALS Point Clouds" (2017) at https://www.mdpi.com/2072-4292/9/9/936. 

## Table of Contents
- Installation
- Usage
- Structure

## Installation

**Warning: this code is still a work in progress, so it's not ready to be used in its fullness right now. Primarily, training data pre-processing needs to be refined. Updates will soon follow! :)** 

To set up the environment and install the required dependencies, it's recommended to use Conda. Follow these steps:
1. **Clone this repository**:

   ```bash
   git clone https://github.com/MatteoFalcioni/MCNN.git
   cd MCNN
   ```

2. **Create a new Conda environment and install dependencies:**:

    ```bash
   conda env create -f environment.yml
   conda activate CNNLidar
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
- `--learning_rate_decay_factor` specifies the learning rate decay factor
- `--momentum` specifies the value of momentum to be used in the scheduler
- `--save_dir` specifies the directory to save trained models

## Structure

- `MCNN/`
  - `data/`
    - `raw`           Unprocessed data
    -  `transforms/` Data pre-processing
       - `point_cloud_to_image.py` Converting raw point cloud data to feature images
  - `models/`
    - `saved/`                Directory in which trained model can be saved
    - `mcnn.py`                Model definitions for MCNN
    - `scnn.py`                 Model definitions for SCNN
  - `results/`                  Directory containing results from the trained models
  - `scripts/`
    - `train.py`                Training and validation functions
  - `tests/`
    - `test_feature_imgs/`      Directory containing feature images produced during testing 
    - `test_models.py`          Unit tests for model functions
    - `test_point_cloud_data_utils.py`     Unit tests for data utilities for the point cloud 
    - `test_point_cloud_to_image`      Unit tests for utilities to convert the point cloud to feature images
    - `test_training.py`        Unit tests for verifying model training
  - `utils/`
    - `config_handler.py`       Configuration management utilities
    - `plot_utils.py`           Plotting utilities
    - `point_cloud_data_utils`  Utilities for point cloud raw data
    - `train_data_utils`        Utilities for the training and validation process
  - `main.py`                   Main script for training the model
  - `config.yaml`              Configuration file with default training settings
  - `environment.yml`          yml file with dependencies required to run the project
  - `README.md`                 This file, providing an overview and instructions





