"""
DenseNet implementation
@File: densenet.py
@Description: DenseNet-121 model definition
"""

import torch
import torch.nn as nn

MODEL_TYPE_DENSENET121 = "densenet121"
MODEL_TYPE_DENSENET169 = "densenet169"
MODEL_TYPE_DENSENET201 = "densenet201"

class BottleNeck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(BottleNeck, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layers, growth_rate, in_planes):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = BottleNeck(in_planes + i * growth_rate, growth_rate)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], dim=1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TransitionLayer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.avgpool(out)
        return out

class DenseNet(nn.Module):
    def __init__(self, block_config, growth_rate, num_classes=1000, init_weights=False):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        # Initial Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense Blocks and Transition Layers
        layers = []
        in_planes = 64
        for i, num_layers in enumerate(block_config):
            # Add Dense Block
            layers.append(DenseBlock(num_layers, growth_rate, in_planes))
            # Calculate output planes after Dense Block
            in_planes = in_planes + num_layers * growth_rate

            # Add Transition Layer (except after the last block)
            if i < len(block_config) - 1:
                # Typically transition layers reduce filters. For DenseNet-121, it's often a fixed ratio or value.
                # Standard DenseNet-121 uses a transition layer that keeps the scale.
                # A common implementation uses out_planes = in_planes // 2 or similar, but
                # let's use a consistent approach: out_planes is calculated based on the growth.
                # For DenseNet-121, transition layers often use 0.5 compression.
                out_planes = in_planes // 2
                layers.append(TransitionLayer(in_planes, out_planes))
                in_planes = out_planes

        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class DenseNet121(DenseNet):
    def __init__(self, num_classes=1000, init_weights=False):
        # DenseNet-121: [6, 12, 24, 16], growth rate 32
        super(DenseNet121, self).__init__(block_config=[6, 12, 24, 16], growth_rate=32, num_classes=num_classes, init_weights=init_weights)

class DenseNet169(DenseNet):
    def __init__(self, num_classes=1000, init_weights=False):
        # DenseNet-169: [6, 12, 32, 32], growth rate 32
        super(DenseNet169, self).__init__(block_config=[6, 12, 32, 32], growth_rate=32, num_classes=num_classes, init_weights=init_weights)

class DenseNet201(DenseNet):
    def __init__(self, num_classes=1000, init_weights=False):
        # DenseNet-201: [6, 12, 48, 32], growth rate 32
        super(DenseNet201, self).__init__(block_config=[6, 12, 48, 32], growth_rate=32, num_classes=num_classes, init_weights=init_weights)
