"""
Transformer-based models for NLP tasks
@File: transformer.py
@Description: Transformer encoder for text classification following PyTorch best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TYPE_TRANSFORMER = "transformer"


class TransformerClassifier(nn.Module):
    """
    Transformer-based text classifier following PyTorch official recommendations
    """
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, 
                 hidden_dim, num_classes, dropout=0.5, max_seq_len=512):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_seq_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers, enable_nested_tensor=False)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights following PyTorch official recommendations"""
        # Initialize embedding with uniform distribution
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize transformer encoder weights
        for name, param in self.transformer_encoder.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    # Xavier uniform for weight matrices
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'norm' in name and 'weight' in name:
                # Layer norm weights initialized to 1
                nn.init.ones_(param)
        
        # Initialize fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def forward(self, x):
        embeds = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)
        embeds = self.pos_encoder(embeds)
        
        output = self.transformer_encoder(embeds)
        
        # Use cls token (first token) for classification
        cls_token = output[:, 0, :]
        cls_token = self.dropout(cls_token)
        logits = self.fc(cls_token)
        
        return logits


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer following official PyTorch implementation
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model) when batch_first=True
        x = x + self.pe[:x.size(1)]  # Use seq_len dimension
        return self.dropout(x)


def transformer_classifier(vocab_size=10000, embedding_dim=256, num_heads=4, 
                           num_layers=3, hidden_dim=512, num_classes=2, 
                           dropout=0.5, max_seq_len=512, **kwargs):
    return TransformerClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        max_seq_len=max_seq_len
    )