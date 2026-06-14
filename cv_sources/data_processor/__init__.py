"""
Data processor module for cv_sources
"""

from .fashion_mnist import DATASET_NAME_FASHION_MNIST, NUM_CLASSES as FM_NUM_CLASSES, load_data_fashion_mnist
from .dogs_vs_cats import DATASET_NAME_DOGS_VS_CATS, NUM_CLASSES as DVC_NUM_CLASSES, load_data_dogs_vs_cats

__all__ = [
    'DATASET_NAME_FASHION_MNIST', 'load_data_fashion_mnist',
    'DATASET_NAME_DOGS_VS_CATS', 'load_data_dogs_vs_cats',
]