"""
MobileNet implementation
@File: mobilenet.py
@Description: MobileNetV1 model definition with depthwise separable convolutions
"""

import torch
import torch.nn as nn

MODEL_TYPE_MOBILENET_1_0 = "mobilenet_1_0"
MODEL_TYPE_MOBILENET_0_5 = "mobilenet_0_5"
MODEL_TYPE_MOBILENET_0_75 = "mobilenet_0_75"


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution consists of:
    1. Depthwise Convolution: applies a single filter per input channel
    2. Pointwise Convolution: 1x1 convolution to combine outputs
    This reduces computation compared to standard convolution.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution: groups=in_channels means each input channel gets its own filter
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        # Pointwise convolution: 1x1 conv to combine depthwise outputs
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x


class MobileNet(nn.Module):
    """
    MobileNetV1 architecture as described in:
    "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    
    Architecture uses depthwise separable convolutions for efficiency.
    Width multiplier (alpha) scales the number of channels.
    """
    def __init__(self, num_classes=1000, width_multiplier=1.0, init_weights=False):
        super(MobileNet, self).__init__()
        
        # Base configuration: (out_channels, stride, n_repeats)
        # First layer is standard conv, rest are depthwise separable
        config = [
            (32, 1, 1),    # Conv2d 3x3
            (64, 1, 1),    # DW + PW
            (128, 2, 1),   # DW + PW with stride 2
            (128, 1, 1),   # DW + PW
            (256, 2, 1),   # DW + PW with stride 2
            (256, 1, 1),   # DW + PW
            (512, 2, 1),   # DW + PW with stride 2
            (512, 1, 5),   # 5x DW + PW repeats
            (1024, 2, 1),  # DW + PW with stride 2
            (1024, 1, 1),  # DW + PW
        ]
        
        # First layer: standard convolution
        first_out = int(32 * width_multiplier)
        self.features = nn.Sequential(
            nn.Conv2d(3, first_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(first_out),
            nn.ReLU(inplace=True)
        )
        
        # Build depthwise separable conv layers
        in_channels = first_out
        for out_channels, stride, n_repeats in config[1:]:
            out_channels = int(out_channels * width_multiplier)
            for i in range(n_repeats):
                # First layer in block uses stride, rest use stride 1
                s = stride if i == 0 else 1
                self.features.add_module(
                    f"ds_{len(self.features)}",
                    DepthwiseSeparableConv(in_channels, out_channels, stride=s)
                )
                in_channels = out_channels
        
        # Average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_channels, num_classes)
        )
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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


def mobilenet_1_0(num_classes=1000, init_weights=False, **kwargs):
    """MobileNetV1 model with 1.0 width multiplier (full size)"""
    return MobileNet(num_classes=num_classes, width_multiplier=1.0, init_weights=init_weights)


def mobilenet_0_5(num_classes=1000, init_weights=False, **kwargs):
    """MobileNet with 0.5 width multiplier (smaller, faster)"""
    return MobileNet(num_classes=num_classes, width_multiplier=0.5, init_weights=init_weights)


def mobilenet_0_75(num_classes=1000, init_weights=False, **kwargs):
    """MobileNet with 0.75 width multiplier"""
    return MobileNet(num_classes=num_classes, width_multiplier=0.75, init_weights=init_weights)