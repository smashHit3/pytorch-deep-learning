"""
Base data processing utilities
@File: base.py
@Description: Shared classes for text classification data processing
"""

from pathlib import Path
import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter


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