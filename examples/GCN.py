import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.conv import GraphConv

from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all
from graphiler.utils import load_data, setup, check_equal, bench, homo_dataset, DEFAULT_DIM
device = setup()


def message_func(edges: EdgeBatchDummy):
    norm = torch.pow(edges.src['degree'], -0.5)
    return {'m': edges.src['h'] * norm}


def reduce_func(nodes: NodeBatchDummy):
    return {'h': torch.relu(torch.sum(nodes.mailbox['m'], dim=1))}


mpdfg = mpdfg_builder(message_func, reduce_func)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.weight = torch.rand(in_dim, out_dim).to(device)

    def forward(self, g, feature, compile=False):
        g.ndata['h'] = torch.mm(feature, self.weight)
        if compile:
            update_all(g, mpdfg)
        else:
            g.update_all(message_func, reduce_func)
        return g.ndata.pop('h')


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()

        self.layer1 = GCNLayer(in_dim, hidden_dim)
        self.layer2 = GCNLayer(out_dim, hidden_dim)

    def forward(self, g, features, compile=False):
        g.ndata['degree'] = g.in_degrees().float().clamp(min=1).unsqueeze(1)
        x = F.relu(self.layer1(g, features, compile))
        x = self.layer2(g, x, compile)
        return x


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


def profile(dataset, feat_dim):
    g, features = load_data(dataset, feat_dim)
    g, features = g.to(device), features.to(device)

    net = GCN(in_dim=features.size()[
              1], hidden_dim=DEFAULT_DIM, out_dim=DEFAULT_DIM).to(device)
    net_dgl = GCN_DGL(in_dim=features.size()[
                      1], hidden_dim=DEFAULT_DIM, out_dim=DEFAULT_DIM).to(device)

    net.eval()
    net_dgl.eval()
    with torch.no_grad():
        bench(net=net_dgl, net_params=(g, features),
              tag="graphconv", nvprof=False, memory=True)
        compile_res = bench(net=net, net_params=(
            g, features, True), tag="compile", nvprof=False, memory=True)
        res = bench(net=net, net_params=(g, features, False),
                    tag="naive", nvprof=False, memory=True)
        check_equal(compile_res, res)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage: python GCN.py [dataset] [feat_dim]")
        exit()
    if sys.argv[1] == "all":
        for d in homo_dataset:
            profile(d, homo_dataset[d])
    else:
        profile(sys.argv[1], int(sys.argv[2]))
