"""
GRU-based model for NLP tasks
@File: gru.py
@Description: GRU implementation for text classification
"""

import torch
import torch.nn as nn

MODEL_TYPE_GRU = "gru"


class GRUClassifier(nn.Module):
    """
    GRU-based text classifier following PyTorch best practices
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
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights following PyTorch official recommendations"""
        # Initialize embedding with uniform distribution
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier uniform initialization
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: orthogonal initialization
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Initialize biases to 0
                nn.init.zeros_(param)
        
        # Initialize fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x):
        embeds = self.embedding(x)
        gru_out, hidden = self.gru(embeds)
        
        # Concatenate last layer's forward and backward hidden states
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        
        return logits


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