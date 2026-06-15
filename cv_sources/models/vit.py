"""
Vision Transformer (ViT) implementation
@File: vit.py
@Description: ViT-Base model definition following "An Image is Worth 16x16 Words"

Key concepts:
1. Image patches: Split image into fixed-size patches
2. Linear embedding: Project patches to embedding dimension
3. Class token: Special token for classification
4. Positional encoding: Add positional information
5. Transformer encoder: Standard encoder blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TYPE_VIT_BASE = "vit_base"
MODEL_TYPE_VIT_SMALL = "vit_small"


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    
    Input: (batch, channels, H, W)
    Output: (batch, num_patches + 1, embed_dim)  # +1 for class token
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolution to extract patches and project to embedding dimension
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Extract patches: (B, embed_dim, num_patches_h, num_patches_w)
        x = self.proj(x)
        
        # Flatten patches: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token: (B, num_patches, embed_dim) -> (B, num_patches + 1, embed_dim)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block:
    LayerNorm -> Multi-Head Attention -> Residual -> LayerNorm -> MLP -> Residual
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) full architecture
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768,
                 num_heads=12, num_layers=12, mlp_ratio=4.0, num_classes=1000,
                 dropout=0.1, init_weights=False):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Transformer encoder
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer encoder
        x = self.encoder(x)
        
        # Classification: take class token output
        x = self.norm(x[:, 0, :])  # (B, embed_dim)
        x = self.head(x)  # (B, num_classes)
        
        return x
    
    def _initialize_weights(self):
        """Initialize weights following ViT paper"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def vit_base(num_classes=1000, init_weights=True, **kwargs):
    """
    ViT-Base model configuration:
    - embed_dim: 768
    - num_heads: 12
    - num_layers: 12
    """
    return ViT(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        num_classes=num_classes,
        init_weights=init_weights,
        **kwargs
    )


def vit_small(num_classes=1000, init_weights=True, **kwargs):
    """
    ViT-Small model configuration:
    - embed_dim: 384
    - num_heads: 6
    - num_layers: 12
    """
    return ViT(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        num_heads=6,
        num_layers=12,
        num_classes=num_classes,
        init_weights=init_weights,
        **kwargs
    )


# Quick test
if __name__ == "__main__":
    model = vit_base(num_classes=10, init_weights=True)
    x = torch.randn(2, 3, 224, 224)  # (batch, channels, H, W)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # Expected: (2, 10)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
