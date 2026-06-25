"""
Text data processing utilities
@File: text_data.py
@Description: Unified data loading interface with backward compatibility
"""

from nlp_sources.data_processor.ag_news import DATASET_NAME_AG_NEWS, load_data_ag_news
from nlp_sources.data_processor.imdb import DATASET_NAME_IMDB, load_data_imdb


DATASET_LOADERS = {
    DATASET_NAME_IMDB: load_data_imdb,
    DATASET_NAME_AG_NEWS: load_data_ag_news,
}


def load_data(dataset, data_root=None, max_samples=None, batch_size=32, max_seq_len=512):
    """
    Load and preprocess text data (unified interface)
    """
    loader = DATASET_LOADERS.get(dataset)
    if loader is None:
        raise ValueError(f"Unknown dataset: {dataset}")
    return loader(data_root, max_samples, batch_size, max_seq_len)