import torch
import torch.nn as nn
from MHCA import MHCA
from QueryBank import QueryBank


class TextAlignmentHead(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.query_bank = QueryBank(dim, num_queries=3)

        self.attn = nn.ModuleList([
            MHCA(dim),
            MHCA(dim),
            MHCA(dim)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, fego):
        """
        fego: [B, C] 或 [B, N, C]
        """

        if fego.dim() == 2:
            fego = fego.unsqueeze(1)  # [B, 1, C]

        B = fego.size(0)

        queries = self.query_bank(B)  # [B, 3, C]

        outputs = []

        for i in range(3):
            q = queries[:, i:i+1, :]  # [B, 1, C]

            q_out = self.attn[i](q, fego, fego)  # [B, 1, C]

            # concat with global fego
            fego_mean = fego.mean(dim=1, keepdim=True)

            combined = torch.cat([q_out, fego_mean], dim=-1)

            out = self.mlp(combined.squeeze(1))  # [B, C]

            outputs.append(out)

        fc_hat, ff_hat, fr_hat = outputs
        return fc_hat, ff_hat, fr_hat