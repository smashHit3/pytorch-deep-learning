"""
Models module for cv_sources
"""

from cv_sources.models.alexnet import AlexNet, MODEL_TYPE_ALEXNET, alexnet
from cv_sources.models.vgg import VGG, vgg11, vgg13, vgg16, vgg19, MODEL_TYPE_VGG11, MODEL_TYPE_VGG13, MODEL_TYPE_VGG16, MODEL_TYPE_VGG19
from cv_sources.models.resnet import ResNet, resnet18, resnet34, resnet50, MODEL_TYPE_RESNET18, MODEL_TYPE_RESNET34, MODEL_TYPE_RESNET50
from cv_sources.models.googlenet import GoogleNet, MODEL_TYPE_GOOGLENET, googlenet
from cv_sources.models.densenet import DenseNet, densenet121, densenet169, densenet201, MODEL_TYPE_DENSENET121, MODEL_TYPE_DENSENET169, MODEL_TYPE_DENSENET201
from cv_sources.models.mobilenet import MobileNet, mobilenet_x1_0, mobilenet_x0_5, mobilenet_x0_75, MODEL_TYPE_MOBILENET_X1_0, MODEL_TYPE_MOBILENET_X0_5, MODEL_TYPE_MOBILENET_X0_75

__all__ = [
    'AlexNet', 'alexnet', 'MODEL_TYPE_ALEXNET',
    'VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MODEL_TYPE_VGG11', 'MODEL_TYPE_VGG13', 'MODEL_TYPE_VGG16', 'MODEL_TYPE_VGG19',
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'MODEL_TYPE_RESNET18', 'MODEL_TYPE_RESNET34', 'MODEL_TYPE_RESNET50',
    'GoogleNet', 'googlenet', 'MODEL_TYPE_GOOGLENET',
    'DenseNet', 'densenet121', 'densenet169', 'densenet201', 'MODEL_TYPE_DENSENET121', 'MODEL_TYPE_DENSENET169', 'MODEL_TYPE_DENSENET201',
    'MobileNet', 'mobilenet_x1_0', 'mobilenet_x0_5', 'mobilenet_x0_75', 'MODEL_TYPE_MOBILENET_X1_0', 'MODEL_TYPE_MOBILENET_X0_5', 'MODEL_TYPE_MOBILENET_X0_75',
]