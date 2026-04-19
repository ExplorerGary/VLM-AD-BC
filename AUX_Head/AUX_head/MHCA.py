import torch
import torch.nn as nn


class MHCA(nn.Module):
    """
    Multi-Head Cross Attention
    q: [B, 1, C]
    k: [B, N, C]
    v: [B, N, C]
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, q, k, v):
        """
        q: [B, 1, C]
        k, v: [B, N, C]
        """
        out, _ = self.attn(q, k, v)
        return out  # [B, 1, C]