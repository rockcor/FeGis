import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv,GCN2Conv,Linear,GATConv

class GCN(Module):
    def __init__(self, num_features,hidden_channels, out_dim,num_layers=2, dropout=0.5):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(GCNConv(num_features, hidden_channels))
        self.lins.append(GCNConv(hidden_channels, out_dim))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers-2):
            self.convs.append(
                GCNConv(hidden_channels,hidden_channels))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[0](x,adj_t).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x,adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x,adj_t)

        return x
class GAT(Module):
    def __init__(self, num_features,hidden_channels, out_dim,num_layers=2, dropout=0.5):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(GATConv(num_features, hidden_channels))
        self.lins.append(GATConv(hidden_channels, out_dim))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers-2):
            self.convs.append(
                GATConv(hidden_channels,hidden_channels))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[0](x,adj_t).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x,adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x,adj_t)

        return x
class GCN2(Module):
    def __init__(self, num_features,hidden_channels, out_dim,num_layers=2, dropout=0.5, alpha=0.1, theta=0.5,
                 shared_weights=False):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_dim))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x
