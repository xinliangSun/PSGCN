import torch
from torch_geometric.nn import GCNConv


class attention_score(torch.nn.Module):
    def __init__(self, in_channels, Conv=GCNConv):
        super(attention_score, self).__init__()
        self.in_channels = in_channels
        self.score_layer = Conv(in_channels, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        score = self.score_layer(x, edge_index)

        return score
