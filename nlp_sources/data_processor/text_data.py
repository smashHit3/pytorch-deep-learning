"""
Text data processing utilities
@File: text_data.py
@Description: Unified data loading interface with backward compatibility
"""

from pathlib import Path
from base import TextDataset, Vocabulary, build_vocab_and_loaders
from imdb import DATASET_NAME_IMDB, load_data_imdb
from ag_news import DATASET_NAME_AG_NEWS, load_data_ag_news


def load_data(text_dataset, data_root=None, max_samples=None, batch_size=32, max_seq_len=512):
    """
    Load and preprocess text data (unified interface)
    """
    if text_dataset == DATASET_NAME_IMDB:
        return load_data_imdb(data_root, max_samples, batch_size, max_seq_len)
    elif text_dataset == DATASET_NAME_AG_NEWS:
        return load_data_ag_news(data_root, max_samples, batch_size, max_seq_len)
    else:
        raise ValueError(f"Unknown dataset: {text_dataset}")