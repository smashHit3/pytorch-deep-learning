"""
Text data processing utilities
@File: text_data.py
@Description: Data loading and preprocessing for text classification
"""

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

DATASET_NAME_IMDB = "imdb"
DATASET_NAME_AG_NEWS = "ag_news"


class TextDataset(Dataset):
    """
    Text classification dataset
    """
    def __init__(self, texts, labels, vocab, max_seq_len=512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        tokens = self.tokenize(text)
        indexed = self.vocab.encode(tokens, self.max_seq_len)
        
        return torch.tensor(indexed, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()


class Vocabulary:
    """
    Vocabulary for text processing
    """
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_count = Counter()
        
    def build(self, texts, min_freq=2):
        for text in texts:
            tokens = self.tokenize(text)
            self.word_count.update(tokens)
            
        for word, count in self.word_count.items():
            if count >= min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()
    
    def encode(self, tokens, max_seq_len):
        indexed = []
        for token in tokens[:max_seq_len]:
            indexed.append(self.word2idx.get(token, 1))
        
        while len(indexed) < max_seq_len:
            indexed.append(0)
            
        return indexed
    
    @property
    def size(self):
        return len(self.word2idx)


def load_imdb_data(data_dir, max_samples=None):
    """
    Load IMDB sentiment analysis dataset
    """
    texts = []
    labels = []
    
    for label, folder in enumerate(['neg', 'pos']):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            continue
            
        for filename in os.listdir(folder_path)[:max_samples]:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(label)
                
    return texts, labels


def load_ag_news_data(data_dir, max_samples=None):
    """
    Load AG News classification dataset
    """
    texts = []
    labels = []
    
    for filename in ['train.csv', 'test.csv']:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                parts = line.strip().split(',', 2)
                if len(parts) >= 3:
                    labels.append(int(parts[0]) - 1)
                    texts.append(parts[2])
                    
    return texts, labels


def load_data(text_dataset, data_root=None, max_samples=None, batch_size=32, max_seq_len=512):
    """
    Load and preprocess text data
    """
    if data_root is None:
        data_root = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    
    if text_dataset == DATASET_NAME_IMDB:
        train_texts, train_labels = load_imdb_data(os.path.join(data_root, 'imdb', 'train'), max_samples)
        test_texts, test_labels = load_imdb_data(os.path.join(data_root, 'imdb', 'test'), max_samples)
        num_classes = 2
    elif text_dataset == DATASET_NAME_AG_NEWS:
        all_texts, all_labels = load_ag_news_data(os.path.join(data_root, 'ag_news'), max_samples)
        split_idx = int(0.8 * len(all_texts))
        train_texts, train_labels = all_texts[:split_idx], all_labels[:split_idx]
        test_texts, test_labels = all_texts[split_idx:], all_labels[split_idx:]
        num_classes = 4
    else:
        raise ValueError(f"Unknown dataset: {text_dataset}")
    
    vocab = Vocabulary()
    vocab.build(train_texts)
    
    train_dataset = TextDataset(train_texts, train_labels, vocab, max_seq_len)
    test_dataset = TextDataset(test_texts, test_labels, vocab, max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, vocab, num_classes