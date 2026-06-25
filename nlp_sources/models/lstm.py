"""
LSTM-based model for NLP tasks
@File: lstm.py
@Description: LSTM implementation for text classification following PyTorch official best practices
"""

import torch
import torch.nn as nn
from typing import Optional
from torch.nn.utils.rnn import pack_padded_sequence

MODEL_TYPE_LSTM = "lstm"


class LSTMClassifier(nn.Module):
    """
    LSTM-based text classifier following PyTorch official best practices.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden states
        num_classes: Number of output classes
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.5)
        bidirectional: Whether to use bidirectional LSTM (default: True)
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5, bidirectional: bool = True,
                 padding_idx: int = 0):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        
        self.init_weights()
        
    def init_weights(self) -> None:
        """Initialize weights following PyTorch official recommendations"""
        # Initialize embedding with uniform distribution
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize LSTM weights following PyTorch official init
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Initialize biases to 0, except forget gate bias to 1 (Jozefowicz et al., 2015)
                nn.init.zeros_(param)
                # LSTM bias layout is [input, forget, cell, output] for each layer/direction.
                gate_size = param.size(0) // 4
                param.data[gate_size:2 * gate_size].fill_(1.0)
        
        # Initialize fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the LSTM classifier.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            lengths: Optional tensor of sequence lengths for proper hidden state selection
        
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        embeds = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        if lengths is None:
            lengths = (x != self.padding_idx).sum(dim=1)
        lengths = lengths.clamp(min=1)

        packed_embeds = pack_padded_sequence(
            embeds,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed_embeds)  # hidden: (num_layers * num_directions, batch_size, hidden_dim)
        
        # Select the appropriate hidden state based on bidirectionality
        if self.bidirectional:
            last_forward = hidden[-2]
            last_backward = hidden[-1]
            last_hidden = torch.cat((last_forward, last_backward), dim=1)
        else:
            last_hidden = hidden[-1]  # Just take the last layer's hidden state
        
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        
        return logits


def lstm_classifier(vocab_size: int = 10000, embedding_dim: int = 128, hidden_dim: int = 256,
                   num_classes: int = 2, num_layers: int = 2, dropout: float = 0.5, 
                   bidirectional: bool = True, padding_idx: int = 0, **kwargs) -> LSTMClassifier:
    """
    Factory function for creating LSTMClassifier instances.
    
    Args:
        vocab_size: Size of vocabulary (default: 10000)
        embedding_dim: Dimension of word embeddings (default: 128)
        hidden_dim: Dimension of LSTM hidden states (default: 256)
        num_classes: Number of output classes (default: 2)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.5)
        bidirectional: Whether to use bidirectional LSTM (default: True)
    
    Returns:
        LSTMClassifier instance
    """
    return LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        padding_idx=padding_idx
    )