import torch
import torch.nn as nn
from MHCA import MHCA
from QueryBank import QueryBank


class ActionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.query_bank = QueryBank(dim, num_queries=3)

        self.attn = nn.ModuleList([
            MHCA(dim),
            MHCA(dim),
            MHCA(dim)
        ])

        self.fc_control = nn.Linear(dim * 2, 4)
        self.fc_turn = nn.Linear(dim * 2, 4)
        self.fc_lane = nn.Linear(dim * 2, 5)

    def forward(self, fego):

        if fego.dim() == 2:
            fego = fego.unsqueeze(1)

        B = fego.size(0)

        queries = self.query_bank(B)

        fego_mean = fego.mean(dim=1, keepdim=True)

        outputs = []

        for i in range(3):
            q = queries[:, i:i+1, :]
            q_out = self.attn[i](q, fego, fego)

            combined = torch.cat([q_out, fego_mean], dim=-1)

            outputs.append(combined.squeeze(1))

        control_hat = self.fc_control(outputs[0])
        turn_hat = self.fc_turn(outputs[1])
        lane_hat = self.fc_lane(outputs[2])

        return control_hat, turn_hat, lane_hat