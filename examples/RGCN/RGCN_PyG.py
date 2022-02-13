import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv, FastRGCNConv


class RGCN_PyG(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels, mode='bmm'):
        super(RGCN_PyG, self).__init__()
        RGCNLayer = FastRGCNConv if mode == 'bmm' else RGCNConv
        self.layer1 = RGCNLayer(in_dim, hidden_dim, num_rels, aggr='add')
        self.layer2 = RGCNLayer(out_dim, hidden_dim, num_rels, aggr='add')

    def forward(self, adj, features, edge_type):
        x = self.layer1(features, adj, edge_type)
        x = F.relu(x)
        x = self.layer2(x, adj, edge_type)
        return x
