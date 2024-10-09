import torch
import torch.nn as nn
from models.scnn import SingleScaleCNN


class MultiScaleCNN(nn.Module):
    """
    A PyTorch implementation of a Multi-Scale Convolutional Neural Network (MCNN) that
    combines the outputs of three Single-Scale CNNs (SCNN1, SCNN2, SCNN3) and performs
    classification through fully connected layers.

    Architecture Overreshape:
    - Inputs: Three n-channel images of size 128x128.
    - Three SCNNs process each input to generate feature maps.
    - Fully connected layers to combine and classify the features from different scales.
    - Output: Classification into 9 classes.
    """

    def __init__(self, channels, classes):
        """
        Initializes the MultiScaleCNN model with three SCNNs and fully connected layers.
        """
        super(MultiScaleCNN, self).__init__()
        
        # Store the number of classes
        self.channels = channels
        self.classes = classes

        # Initialize three SCNNs
        self.scnn1 = SingleScaleCNN(channels=channels)
        self.scnn2 = SingleScaleCNN(channels=channels)
        self.scnn3 = SingleScaleCNN(channels=channels)

        # First MCNN Layer (FC + BN + ReLU) to combine SCNN outputs
        self.fc_fusion = nn.Linear(8 * 8 * 128 * 3, 4096)  # Combining outputs from three SCNNs
        self.bn_fusion = nn.BatchNorm1d(4096)
        self.relu_fusion = nn.ReLU()

        # Second MCNN layer (again FC + BN + ReLu)
        self.fc1 = nn.Linear(4096, 4096)
        self.bn_fc1 = nn.BatchNorm1d(4096)
        self.relu_fc1 = nn.ReLU()

        # Final Fully Connected Layer for Output
        self.fc2 = nn.Linear(4096, classes)  # Output layer with a chosen number of classes

    def forward(self, x1, x2, x3):
        """
        Defines the forward pass of the MultiScaleCNN model.

        Args:
            x1 (torch.Tensor): Input tensor for SCNN1 of shape (batch_size, channels, 128, 128).
            x2 (torch.Tensor): Input tensor for SCNN2 of shape (batch_size, channels, 128, 128).
            x3 (torch.Tensor): Input tensor for SCNN3 of shape (batch_size, channels, 128, 128).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, classes) representing the class scores.
        """
        # Forward pass through each SCNN
        out1 = self.scnn1(x1)  # Output: (batch_size, 128, 8, 8)
        out2 = self.scnn2(x2)  # Output: (batch_size, 128, 8, 8)
        out3 = self.scnn3(x3)  # Output: (batch_size, 128, 8, 8)

        # Need to fuse together the outputs: we flatten + concatenate
        # Flatten the outputs
        out1 = out1.reshape(out1.size(0), -1)  # Flatten: (batch_size, 128 * 8 * 8)
        out2 = out2.reshape(out2.size(0), -1)  # Flatten: (batch_size, 128 * 8 * 8)
        out3 = out3.reshape(out3.size(0), -1)  # Flatten: (batch_size, 128 * 8 * 8)

        # Concatenate the flattened outputs
        combined = torch.cat((out1, out2, out3), dim=1)  # Combined: (batch_size, 3 * 128 * 8 * 8), 3 becuse of 3 scnn's

        # First FC Layer + BatchNorm + ReLU
        x = self.fc_fusion(combined)
        x = self.bn_fusion(x)
        x = self.relu_fusion(x)

        # Second FC Layer + BatchNorm + ReLU
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)

        # Final FC Layer to produce class scores
        x = self.fc2(x)  # Output layer for classes: (batch_size, classes)

        return x


