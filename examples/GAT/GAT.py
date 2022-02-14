import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch_sparse import SparseTensor

from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all
from graphiler.utils import load_data, setup, check_equal, bench, homo_dataset, DEFAULT_DIM, init_log

from GAT_DGL import GAT_DGL
from GAT_PyG import GAT_PyG


device = setup()


# pass extra parameters as a workaround
def message_func(edges: EdgeBatchDummy, fc_weight, attn_weight):
    z_s = torch.mm(edges.src['h'], fc_weight)
    z_d = torch.mm(edges.dst['h'], fc_weight)
    z2 = torch.cat([z_s, z_d], dim=1)
    a = torch.mm(z2, attn_weight)
    # Todo: F.leaky_relu
    return {'z': z_s, 'e': torch.relu(a)}


def reduce_func(nodes: NodeBatchDummy):
    alpha = torch.softmax(nodes.mailbox['e'], dim=1)
    h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    return {'h': h}


mpdfg = mpdfg_builder(message_func, reduce_func)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc_weight = torch.rand(in_dim, out_dim).to(device)
        self.attn_weight = torch.rand(2 * out_dim, 1).to(device)

    def message_func(self, edges):
        z_s = torch.mm(edges.src['h'], self.fc_weight)
        z_d = torch.mm(edges.dst['h'], self.fc_weight)
        z2 = torch.cat([z_s, z_d], dim=1)
        a = torch.mm(z2, self.attn_weight)
        return {'z': z_s, 'e': torch.relu(a)}

    def forward(self, g, feature, compile=False):
        g.ndata['h'] = feature
        if compile:
            update_all(g, mpdfg, msg_params=(
                self.fc_weight, self.attn_weight))
        else:
            g.update_all(self.message_func, reduce_func)
        return g.ndata.pop('h')


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GAT, self).__init__()
        self.layer1 = GATLayer(in_dim, hidden_dim)
        self.layer2 = GATLayer(hidden_dim, out_dim)

    def forward(self, g, features, compile=False):
        h = self.layer1(g, features, compile)
        h = F.elu(h)
        h = self.layer2(g, h, compile)
        return h


def profile(dataset, feat_dim, steps=1000):
    log = init_log(['DGL-primitives', 'PyG-primitives', 'Graphiler', 'DGL-UDF'], ['time', 'mem'])
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim)
    features = features.to(device)

    def run_baseline_graphiler(g, features):
        g = g.to(device)
        net = GAT(in_dim=feat_dim, hidden_dim=DEFAULT_DIM,
                  out_dim=DEFAULT_DIM).to(device)
        net.eval()
        with torch.no_grad():
            compile_res = bench(net=net, net_params=(
                g, features, True), tag="Graphiler", nvprof=False, steps=steps, memory=True, log=log)
            res = bench(net=net, net_params=(g, features, False),
                        tag="DGL-UDF", nvprof=False, steps=steps, memory=True, log=log)
            check_equal(compile_res, res)
        del g, net, compile_res, res

    def run_pyg(g, features):
        u, v = g.edges()
        adj = SparseTensor(row=u, col=v, sparse_sizes=(
            g.num_src_nodes(), g.num_dst_nodes())).to(device)
        net_pyg = GAT_PyG(in_dim=feat_dim, hidden_dim=DEFAULT_DIM,
                          out_dim=DEFAULT_DIM).to(device)
        net_pyg.eval()
        with torch.no_grad():
            bench(net=net_pyg, net_params=(features, adj),
                  tag="PyG-primitives", nvprof=False, steps=steps, memory=True, log=log)
        del u, v, adj, net_pyg

    def run_dgl(g, features):
        g = g.to(device)
        net_dgl = GAT_DGL(in_dim=feat_dim, hidden_dim=DEFAULT_DIM,
                          out_dim=DEFAULT_DIM).to(device)
        net_dgl.eval()
        with torch.no_grad():
            bench(net=net_dgl, net_params=(g, features),
                  tag="DGL-primitives", nvprof=False, steps=steps, memory=True, log=log)
        del g, net_dgl

    run_baseline_graphiler(g, features)
    run_pyg(g, features)
    run_dgl(g, features)

    return log


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage: python GAT.py [dataset] [feat_dim]")
        exit()
    if sys.argv[1] == "all":
        log = {}
        for d in homo_dataset:
            log[d] = profile(d, homo_dataset[d])
        pd.DataFrame(log).to_pickle("./GAT.pkl")
    else:
        profile(sys.argv[1], int(sys.argv[2]))
