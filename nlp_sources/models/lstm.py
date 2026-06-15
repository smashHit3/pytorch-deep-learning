"""
LSTM-based model for NLP tasks
@File: lstm.py
@Description: LSTM implementation for text classification
"""

import torch
import torch.nn as nn

MODEL_TYPE_LSTM = "lstm"


class LSTMClassifier(nn.Module):
    """
    LSTM-based text classifier following PyTorch best practices
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
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights following PyTorch official recommendations"""
        # Initialize embedding with uniform distribution
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: uniform initialization
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: orthogonal initialization
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Initialize biases to 0, except forget gate bias to 1
                nn.init.zeros_(param)
                # Forget gate bias indices (bias_f): [hidden_dim:2*hidden_dim] for each layer
                hidden_dim = param.size(0) // 4  # LSTM has 4 gates
                for i in range(self.lstm.num_layers):
                    start = i * 4 * hidden_dim + hidden_dim
                    end = start + hidden_dim
                    param.data[start:end].fill_(1.0)
        
        # Initialize fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embeds)
        
        # Concatenate last layer's forward and backward hidden states
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
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