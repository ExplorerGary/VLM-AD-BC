import torch

import torch.nn as nn

class QueryBank(nn.Module):
    """
    管理 learnable queries
    """
    def __init__(self, dim, num_queries=3):
        super().__init__()
        self.queries = nn.Parameter(
        torch.randn(num_queries, dim)
        )



    def forward(self, batch_size):
        """
        返回 [B, num_queries, C]
        """

        return self.queries.unsqueeze(0).expand(batch_size, -1, -1)