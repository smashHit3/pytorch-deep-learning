"""
Models module for cv_sources
"""

from .alexnet import AlexNet, MODEL_TYPE_ALEXNET
from .vgg import VGG, vgg11, vgg13, vgg16, vgg19, MODEL_TYPE_VGG11, MODEL_TYPE_VGG13, MODEL_TYPE_VGG16, MODEL_TYPE_VGG19
from .resnet import ResNet, ResNet18, ResNet34, ResNet50, MODEL_TYPE_RESNET18, MODEL_TYPE_RESNET34, MODEL_TYPE_RESNET50
from .googlenet import GoogleNet, MODEL_TYPE_GOOGLENET
from .densenet import DenseNet, DenseNet121, DenseNet169, DenseNet201, MODEL_TYPE_DENSENET121, MODEL_TYPE_DENSENET169, MODEL_TYPE_DENSENET201
from .mobilenet import MobileNet, mobilenet_v1, mobilenet_0_5, mobilenet_0_75, MODEL_TYPE_MOBILENET, MODEL_TYPE_MOBILENET_0_5, MODEL_TYPE_MOBILENET_0_75

__all__ = [
    'AlexNet', 'MODEL_TYPE_ALEXNET',
    'VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MODEL_TYPE_VGG11', 'MODEL_TYPE_VGG13', 'MODEL_TYPE_VGG16', 'MODEL_TYPE_VGG19',
    'ResNet', 'ResNet18', 'ResNet34', 'ResNet50', 'MODEL_TYPE_RESNET18', 'MODEL_TYPE_RESNET34', 'MODEL_TYPE_RESNET50',
    'GoogleNet', 'MODEL_TYPE_GOOGLENET',
    'DenseNet', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'MODEL_TYPE_DENSENET121', 'MODEL_TYPE_DENSENET169', 'MODEL_TYPE_DENSENET201',
    'MobileNet', 'mobilenet_v1', 'mobilenet_0_5', 'mobilenet_0_75', 'MODEL_TYPE_MOBILENET', 'MODEL_TYPE_MOBILENET_0_5', 'MODEL_TYPE_MOBILENET_0_75',
]