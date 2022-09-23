import math
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, Conv1d
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.utils import dropout_adj
from layers import attention_score
from util_functions import *


class PSGCN(torch.nn.Module):
    def __init__(self, dataset, gconv=GCNConv, latent_dim=[64, 64, 1], k=30,
                 dropout=0.3, force_undirected=False):
        super(PSGCN, self).__init__()

        self.dropout = dropout
        self.force_undirected = force_undirected
        self.score1 = attention_score(latent_dim[0])
        self.score2 = attention_score(latent_dim[1])
        self.score3 = attention_score(latent_dim[2])

        self.conv1 = gconv(dataset.num_features, latent_dim[0])
        self.conv2 = gconv(latent_dim[0], latent_dim[1])
        self.conv3 = gconv(latent_dim[1], latent_dim[2])

        if k < 1:  # transform percentile to number
            node_nums = sorted([g.num_nodes for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums)))-1]
            k = max(10, k)  # no smaller than 10

        self.k = int(k)
        self.dropout = dropout
        conv1d_channels = [16, 32]
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)

        self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(self.dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # delete edge_attribute
        edge_index, edge_type = dropout_adj(
            edge_index, p=self.dropout,
            force_undirected=self.force_undirected, num_nodes=len(x),
            training=self.training
        )

        x = torch.relu(self.conv1(x, edge_index))
        attention_score1 = self.score1(x, edge_index)
        x1 = torch.mul(attention_score1, x)

        x = torch.relu(self.conv2(x, edge_index))
        attention_score2 = self.score2(x, edge_index)
        x2 = torch.mul(attention_score2, x)

        x = torch.relu((self.conv3(x, edge_index)))
        attention_score3 = self.score3(x, edge_index)
        x3 = torch.mul(attention_score3, x)

        X = [x1, x2, x3]
        concat_states = torch.cat(X, 1)

        x = global_sort_pool(concat_states, batch, self.k)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1d_params1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_params2(x))
        x = x.view(len(x), -1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x[:, 0]



