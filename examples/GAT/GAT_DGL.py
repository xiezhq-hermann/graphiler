import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.conv import GATConv


class GAT_DGL(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GAT_DGL, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim,
                              num_heads=1, allow_zero_in_degree=True)
        self.layer2 = GATConv(hidden_dim, out_dim,
                              num_heads=1, allow_zero_in_degree=True)

    def forward(self, g, features):
        h = self.layer1(g, features)
        h = F.elu(h)
        h = self.layer2(g, h)
        return h
