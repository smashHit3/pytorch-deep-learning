"""
Models module for cv_sources
"""

from cv_sources.models.alexnet import AlexNet, MODEL_TYPE_ALEXNET, create_alexnet
from cv_sources.models.vgg import VGG, create_vgg11, create_vgg13, create_vgg16, create_vgg19, MODEL_TYPE_VGG11, MODEL_TYPE_VGG13, MODEL_TYPE_VGG16, MODEL_TYPE_VGG19
from cv_sources.models.resnet import ResNet, create_resnet18, create_resnet34, create_resnet50, MODEL_TYPE_RESNET18, MODEL_TYPE_RESNET34, MODEL_TYPE_RESNET50
from cv_sources.models.googlenet import GoogleNet, MODEL_TYPE_GOOGLENET, create_googlenet
from cv_sources.models.densenet import DenseNet, create_densenet121, create_densenet169, create_densenet201, MODEL_TYPE_DENSENET121, MODEL_TYPE_DENSENET169, MODEL_TYPE_DENSENET201
from cv_sources.models.mobilenet import MobileNet, create_mobilenet_x1_0, create_mobilenet_x0_5, create_mobilenet_x0_75, MODEL_TYPE_MOBILENET_X1_0, MODEL_TYPE_MOBILENET_X0_5, MODEL_TYPE_MOBILENET_X0_75

__all__ = [
    'AlexNet', 'create_alexnet', 'MODEL_TYPE_ALEXNET',
    'VGG', 'create_vgg11', 'create_vgg13', 'create_vgg16', 'create_vgg19', 'MODEL_TYPE_VGG11', 'MODEL_TYPE_VGG13', 'MODEL_TYPE_VGG16', 'MODEL_TYPE_VGG19',
    'ResNet', 'create_resnet18', 'create_resnet34', 'create_resnet50', 'MODEL_TYPE_RESNET18', 'MODEL_TYPE_RESNET34', 'MODEL_TYPE_RESNET50',
    'GoogleNet', 'create_googlenet', 'MODEL_TYPE_GOOGLENET',
    'DenseNet', 'create_densenet121', 'create_densenet169', 'create_densenet201', 'MODEL_TYPE_DENSENET121', 'MODEL_TYPE_DENSENET169', 'MODEL_TYPE_DENSENET201',
    'MobileNet', 'create_mobilenet_x1_0', 'create_mobilenet_x0_5', 'create_mobilenet_x0_75', 'MODEL_TYPE_MOBILENET_X1_0', 'MODEL_TYPE_MOBILENET_X0_5', 'MODEL_TYPE_MOBILENET_X0_75',
]