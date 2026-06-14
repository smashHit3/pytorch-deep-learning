"""
Data processor module for nlp_sources
"""

from nlp_sources.data_processor.base import TextDataset, Vocabulary
from nlp_sources.data_processor.imdb import DATASET_NAME_IMDB, load_data_imdb
from nlp_sources.data_processor.ag_news import DATASET_NAME_AG_NEWS, load_data_ag_news
from nlp_sources.data_processor.text_data import load_data

__all__ = [
    'TextDataset', 'Vocabulary', 'load_data',
    'DATASET_NAME_IMDB', 'DATASET_NAME_AG_NEWS',
    'load_data_imdb', 'load_data_ag_news'
]