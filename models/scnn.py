import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleScaleCNN(nn.Module):
    """
    A PyTorch implementation of a Single-Scale Convolutional Neural Network (SCNN) designed
    for processing input images at a single scale. This model extracts hierarchical features
    through a series of convolutional layers with batch normalization and ReLU activation,
    followed by max-pooling layers to downsample the spatial dimensions.

    Architecture Overview:
    - Input: n-channel image (e.g., 3 for RGB) of size 128x128.
    - Five sequential convolutional blocks:
      - Each block (except for the 4th*) consists of:
        - A 3x3 Convolutional layer with padding of 1.
        - A Batch Normalization layer.
        - A ReLU activation function.
      - Max pooling layers are applied after the first, second, third, and fifth convolutional layers
        to downsample the spatial dimensions by half.
      * 4th block consists only of Batch Normalization and ReLu
    - Output: A feature map of size 8x8x128.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with 32 output channels and a 3x3 kernel.
        bn1 (nn.BatchNorm2d): Batch normalization for the first convolutional layer.
        pool1 (nn.MaxPool2d): Max pooling layer with a 2x2 kernel after the first convolutional block.

        conv2 (nn.Conv2d): Second convolutional layer with 64 output channels and a 3x3 kernel.
        bn2 (nn.BatchNorm2d): Batch normalization for the second convolutional layer.
        pool2 (nn.MaxPool2d): Max pooling layer with a 2x2 kernel after the second convolutional block.

        conv3 (nn.Conv2d): Third convolutional layer with 128 output channels and a 3x3 kernel.
        bn3 (nn.BatchNorm2d): Batch normalization for the third convolutional layer.
        pool3 (nn.MaxPool2d): Max pooling layer with a 2x2 kernel after the third convolutional block.

        conv4 (nn.Conv2d): Fourth convolutional layer with 128 output channels and a 3x3 kernel.
        bn4 (nn.BatchNorm2d): Batch normalization for the fourth convolutional layer.

        conv5 (nn.Conv2d): Fifth convolutional layer with 128 output channels and a 3x3 kernel.
        bn5 (nn.BatchNorm2d): Batch normalization for the fifth convolutional layer.
        pool5 (nn.MaxPool2d): Max pooling layer with a 2x2 kernel after the fifth convolutional block.
    """

    def __init__(self, channels=3):
        """
        Initializes the SingleScaleCNN model with its layers.
        """
        super(SingleScaleCNN, self).__init__()

        # First Convolutional Block: nx128x128 --> 32x64x64  (recall: in Torch, channel dimension goes first)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, padding=1)  # Output: 32x128x128
        self.bn1 = nn.BatchNorm2d(32)   # Batch normalization doesn't change the spatial dimension
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32x64x64 (MaxPool changes spatial dimension)

        # Second Convolutional Block: 32x64x64 --> 64x32x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Output: 64x64x64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x32x32

        # Third Convolutional Block: 64x32x32 --> 128x16x16
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # Output: 128x32x32
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128x16x16

        # Fourth Convolutional Block: 128x16x16 unchanged here (no max pooling)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # Output: 128x16x16
        self.bn4 = nn.BatchNorm2d(128)  # Output: 128x16x16

        # Fifth and final Convolutional Block: 128x16x16 --> 128x8x8
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # Output: 128x16x16
        self.bn5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128x8x8

    def forward(self, x):
        """
        Defines the forward pass of the SingleScaleCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 128, 128) representing a batch of images.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 128, 8, 8) representing the feature map.
        """
        # First Convolutional Block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # Output: 32x64x64

        # Second Convolutional Block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # Output: 64x32x32

        # Third Convolutional Block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # Output: 128x16x16

        # Fourth Convolutional Block
        x = F.relu(self.bn4(self.conv4(x)))  # Output: 128x16x16 (No Max Pooling)

        # Fifth Convolutional Block
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))  # Output: 128x8x8

        return x
