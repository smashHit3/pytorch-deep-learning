"""
Base data processing utilities
@File: base.py
@Description: Shared classes for text classification data processing
"""

import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PAD_IDX = 0
UNK_IDX = 1


def _normalize_and_tokenize(text: str) -> list[str]:
    normalized = text.lower()
    normalized = re.sub(r'[^a-zA-Z0-9\s]', '', normalized)
    return normalized.split()


class TextDataset(Dataset):
    """
    Text classification dataset
    """
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: 'Vocabulary', max_seq_len: int = 512):
        if len(texts) != len(labels):
            raise ValueError('texts and labels must have the same length')
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        
        tokens = self.tokenize(text)
        indexed = self.vocab.encode(tokens, self.max_seq_len)
        
        return torch.tensor(indexed, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    @staticmethod
    def tokenize(text: str) -> list[str]:
        return _normalize_and_tokenize(text)


class Vocabulary:
    """
    Vocabulary for text processing
    """
    def __init__(self):
        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.word_count = Counter()
        
    def build(self, texts: Sequence[str], min_freq: int = 2) -> None:
        for text in texts:
            tokens = self.tokenize(text)
            self.word_count.update(tokens)
            
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                
    @staticmethod
    def tokenize(text: str) -> list[str]:
        return _normalize_and_tokenize(text)
    
    def encode(self, tokens: Sequence[str], max_seq_len: int) -> list[int]:
        indexed = [self.word2idx.get(token, UNK_IDX) for token in tokens[:max_seq_len]]
        if len(indexed) < max_seq_len:
            indexed.extend([PAD_IDX] * (max_seq_len - len(indexed)))
        return indexed
    
    def save(self, path: str | Path) -> None:
        """Save vocabulary to file"""
        import json

        path = Path(path)
        data = {
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()}
        }
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> 'Vocabulary':
        """Load vocabulary from file"""
        import json

        path = Path(path)
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = cls()
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        return vocab

    @property
    def size(self):
        return len(self.word2idx)


def build_vocab_and_loaders(train_texts, train_labels, test_texts, test_labels, 
                            batch_size=32, max_seq_len=512):
    """
    Build vocabulary and create data loaders
    """
    vocab = Vocabulary()
    vocab.build(train_texts)
    
    train_dataset = TextDataset(train_texts, train_labels, vocab, max_seq_len)
    test_dataset = TextDataset(test_texts, test_labels, vocab, max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, vocab