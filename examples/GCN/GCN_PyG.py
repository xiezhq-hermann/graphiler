import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class GCN_PyG(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super(GCN_PyG, self).__init__()

        self.layer1 = GCNConv(in_dim, hidden_dim, cached=False)
        self.layer2 = GCNConv(hidden_dim, out_dim, cached=False)

    def forward(self, x, adj):
        h = self.layer1(x, adj)
        h = F.relu(h)
        h = self.layer2(h, adj)
        return h
