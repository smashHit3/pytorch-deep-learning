"""
Data processor module for nlp_sources
"""

from .base import TextDataset, Vocabulary
from .imdb import DATASET_NAME_IMDB, load_data_imdb
from .ag_news import DATASET_NAME_AG_NEWS, load_data_ag_news
from .text_data import load_data

__all__ = [
    'TextDataset', 'Vocabulary', 'load_data',
    'DATASET_NAME_IMDB', 'DATASET_NAME_AG_NEWS',
    'load_data_imdb', 'load_data_ag_news'
]