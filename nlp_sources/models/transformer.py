"""
Transformer-based models for NLP tasks
@File: transformer.py
@Description: Transformer encoder for text classification following PyTorch official best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

MODEL_TYPE_TRANSFORMER = "transformer"


class TransformerClassifier(nn.Module):
    """
    Transformer-based text classifier following PyTorch official recommendations.
    
    This implementation follows the official PyTorch transformer implementation
    and includes best practices for text classification tasks.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings (d_model)
        num_heads: Number of attention heads
        num_layers: Number of transformer encoder layers
        hidden_dim: Dimension of feedforward network
        num_classes: Number of output classes
        dropout: Dropout probability (default: 0.5)
        max_seq_len: Maximum sequence length (default: 512)
        activation: Activation function in feedforward network (default: 'gelu')
        layer_norm_eps: Epsilon for layer normalization (default: 1e-5)
    """
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int, num_layers: int,
                 hidden_dim: int, num_classes: int, dropout: float = 0.5, max_seq_len: int = 512,
                 activation: str = 'gelu', layer_norm_eps: float = 1e-5, padding_idx: int = 0):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_seq_len)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=True  # Pre-norm architecture, better for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        # Classification head with pre-classifier layer (BERT-style)
        self.pre_classifier = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
        self.init_weights()
        
    def init_weights(self) -> None:
        """Initialize weights following PyTorch official recommendations"""
        # Initialize embedding with uniform distribution (similar to BERT)
        nn.init.uniform_(self.embedding.weight, -0.02, 0.02)
        
        # Initialize transformer encoder weights following official PyTorch init
        for name, param in self.transformer_encoder.named_parameters():
            if 'weight' in name and param.dim() > 1:
                # Xavier uniform for weight matrices
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize pre-classifier and classifier layers
        nn.init.xavier_uniform_(self.pre_classifier.weight)
        nn.init.zeros_(self.pre_classifier.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
    def _generate_padding_mask(self, x: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
        """
        Generate padding mask for transformer encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            padding_idx: Index used for padding (default: 0)
        
        Returns:
            mask: Boolean tensor of shape (batch_size, seq_len) where True indicates padding
        """
        return (x == padding_idx)
    
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Transformer classifier.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            padding_mask: Optional boolean tensor of shape (batch_size, seq_len) 
                         where True indicates padding positions
        
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        # Generate padding mask if not provided
        if padding_mask is None:
            padding_mask = self._generate_padding_mask(x)
        
        # Embedding layer with scaling
        embeds = self.embedding(x) * math.sqrt(self.embedding_dim)
        embeds = self.pos_encoder(embeds)
        
        # Transformer encoder forward pass
        output = self.transformer_encoder(embeds, src_key_padding_mask=padding_mask)
        
        # Use mask-aware mean pooling over valid tokens.
        valid_mask = (~padding_mask).unsqueeze(-1).to(output.dtype)
        pooled = (output * valid_mask).sum(dim=1)
        token_count = valid_mask.sum(dim=1).clamp(min=1.0)
        pooled = pooled / token_count
        
        # Apply pre-classifier layer with GELU activation (BERT-style)
        pooled = self.pre_classifier(pooled)
        pooled = F.gelu(pooled)
        pooled = self.dropout(pooled)
        
        # Final classification layer
        logits = self.fc(pooled)
        
        return logits


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer following official PyTorch implementation.
    
    Implements the sinusoidal positional encoding described in:
    "Attention is All You Need" (Vaswani et al., 2017)
    
    Args:
        d_model: Dimension of the model
        dropout: Dropout probability (default: 0.1)
        max_len: Maximum sequence length (default: 5000)
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        # Shape: (1, max_len, d_model) for batch_first inputs.
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model) when batch_first=True
        
        Returns:
            Tensor with positional encoding added
        """
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Input sequence length {x.size(1)} exceeds maximum positional encoding length {self.pe.size(1)}"
            )
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def transformer_classifier(vocab_size: int = 10000, embedding_dim: int = 128, num_heads: int = 4,
                           num_layers: int = 3, hidden_dim: int = 256, num_classes: int = 2,
                           dropout: float = 0.5, max_seq_len: int = 512, padding_idx: int = 0,
                           **kwargs) -> TransformerClassifier:
    """
    Factory function for creating TransformerClassifier instances.
    
    Args:
        vocab_size: Size of vocabulary (default: 10000)
        embedding_dim: Dimension of word embeddings (d_model) (default: 128)
        num_heads: Number of attention heads (default: 4)
        num_layers: Number of transformer encoder layers (default: 3)
        hidden_dim: Dimension of feedforward network (default: 256)
        num_classes: Number of output classes (default: 2)
        dropout: Dropout probability (default: 0.5)
        max_seq_len: Maximum sequence length (default: 512)
    
    Returns:
        TransformerClassifier instance
    """
    return TransformerClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        max_seq_len=max_seq_len,
        padding_idx=padding_idx
    )