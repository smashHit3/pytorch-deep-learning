"""
Data processor module for nlp_sources
"""

from .text_data import (
    TextDataset, Vocabulary, load_data, 
    DATASET_NAME_IMDB, DATASET_NAME_AG_NEWS
)

__all__ = [
    'TextDataset', 'Vocabulary', 'load_data',
    'DATASET_NAME_IMDB', 'DATASET_NAME_AG_NEWS'
]