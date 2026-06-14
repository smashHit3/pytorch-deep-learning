"""
AlexNet implementation
@File: alex_net.py
@Description: AlexNet model definition
"""

import torch
import torch.nn as nn

MODEL_TYPE_ALEXNET = "alexnet"

class AlexNet(nn.Module):
    def __init__(self, num_classes : int = 1000, dropout : float = 0.5, init_weights=False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # Convolution layer 1
            nn.ReLU(inplace=True),  # Activation function 1
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pooling layer 1
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # Convolution layer 2
            nn.ReLU(inplace=True),  # Activation function 2
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pooling layer 2
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # Convolution layer 3
            nn.ReLU(inplace=True),  # Activation function 3
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Convolution layer 4
            nn.ReLU(inplace=True),  # Activation function 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Convolution layer 5
            nn.ReLU(inplace=True),  # Activation function 5
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pooling layer 3
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # Adaptive average pooling layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),  # Dropout layer 1
            nn.Linear(256 * 6 * 6, 4096),  # Fully connected layer
            nn.ReLU(inplace=True),  # Activation function 6
            nn.Dropout(p=dropout),  # Dropout layer 2
            nn.Linear(4096, 4096),  # Fully connected layer
            nn.ReLU(inplace=True),  # Activation function 7
            nn.Linear(4096, num_classes),  # Fully connected layer
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # Extract features through convolutional layers
        x = self.avgpool(x)  # Adjust feature map size through adaptive average pooling
        x = torch.flatten(x, 1)  # Flatten feature map into vector
        x = self.classifier(x)  # Classify through fully connected layers
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def alexnet(num_classes=1000, dropout=0.5, init_weights=True, **kwargs):
    """AlexNet model"""
    return AlexNet(num_classes=num_classes, dropout=dropout, init_weights=init_weights)
