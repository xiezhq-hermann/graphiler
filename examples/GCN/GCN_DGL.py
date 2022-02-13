import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.conv import GraphConv


class GCN_DGL(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN_DGL, self).__init__()
        self.layer1 = GraphConv(in_dim, hidden_dim, norm='right',
                                allow_zero_in_degree=True, activation=torch.relu)
        self.layer2 = GraphConv(hidden_dim, out_dim, norm='right',
                                allow_zero_in_degree=True, activation=torch.relu)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x
