from action_alignment_head import ActionHead
from text_alignment_head import TextAlignmentHead
from AUX_loss import AUX_loss
import torch.nn as nn



class AUXHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.action_head = ActionHead(dim)
        self.text_head = TextAlignmentHead(dim)

    def forward(self, fego):
        control_hat, turn_hat, lane_hat = self.action_head(fego)
        fc_hat, ff_hat, fr_hat = self.text_head(fego)

        info =  {
            'control_hat': control_hat,
            'turn_hat': turn_hat,
            'lane_hat': lane_hat,
            'fc_hat': fc_hat,
            'ff_hat': ff_hat,
            'fr_hat': fr_hat
        }
        return info