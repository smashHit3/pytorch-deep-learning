"""
RNN-based models for NLP tasks
@File: rnn.py
@Description: LSTM and GRU implementations for text classification
"""

import torch
import torch.nn as nn

MODEL_TYPE_LSTM = "lstm"
MODEL_TYPE_GRU = "gru"


class LSTMClassifier(nn.Module):
    """
    LSTM-based text classifier
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, 
                 num_layers=2, dropout=0.5, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        
        return logits


class GRUClassifier(nn.Module):
    """
    GRU-based text classifier
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.5, bidirectional=True):
        super(GRUClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        
    def forward(self, x):
        embeds = self.embedding(x)
        gru_out, _ = self.gru(embeds)
        
        last_hidden = gru_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        
        return logits


def lstm_classifier(vocab_size=10000, embedding_dim=128, hidden_dim=256, 
                   num_classes=2, num_layers=2, dropout=0.5, bidirectional=True, **kwargs):
    return LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )


def gru_classifier(vocab_size=10000, embedding_dim=128, hidden_dim=256,
                  num_classes=2, num_layers=2, dropout=0.5, bidirectional=True, **kwargs):
    return GRUClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )