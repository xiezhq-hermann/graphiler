import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

from torch_sparse import SparseTensor

from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all
from graphiler.utils import load_data, setup, check_equal, bench, homo_dataset, DEFAULT_DIM, init_log, empty_cache

from GCN_DGL import GCN_DGL
from GCN_PyG import GCN_PyG

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


def profile(dataset, feat_dim, repeat=1000):
    log = init_log(["DGL-primitives", "PyG-primitives",
                    "Graphiler", "DGL-UDF"], ["time", "mem"])
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim, prepare=False)
    features = features.to(device)

    @empty_cache
    def run_baseline_and_graphiler(g, features):
        g, _ = load_data(dataset, feat_dim, prepare=True)
        g = g.to(device)
        net = GCN(in_dim=feat_dim, hidden_dim=DEFAULT_DIM,
                  out_dim=DEFAULT_DIM).to(device)
        net.eval()
        with torch.no_grad():
            compile_res = bench(net=net, net_params=(
                g, features, True), tag="Graphiler", nvprof=False, repeat=repeat, memory=True, log=log)
            res = bench(net=net, net_params=(g, features, False),
                        tag="DGL-UDF", nvprof=False, repeat=repeat, memory=True, log=log)
            check_equal(compile_res, res)
        del g, net, compile_res, res

    @empty_cache
    def run_pyg(g, features):
        u, v = g.edges()
        adj = SparseTensor(row=u, col=v, sparse_sizes=(
            g.num_src_nodes(), g.num_dst_nodes())).to(device)
        net_pyg = GCN_PyG(in_dim=feat_dim, hidden_dim=DEFAULT_DIM,
                          out_dim=DEFAULT_DIM).to(device)
        net_pyg.eval()
        with torch.no_grad():
            bench(net=net_pyg, net_params=(features, adj),
                  tag="PyG-primitives", nvprof=False, repeat=repeat, memory=True, log=log)
        return u, v, adj, net_pyg

    @empty_cache
    def run_dgl(g, features):
        g = g.to(device)
        net_dgl = GCN_DGL(in_dim=feat_dim, hidden_dim=DEFAULT_DIM,
                          out_dim=DEFAULT_DIM).to(device)
        net_dgl.eval()
        with torch.no_grad():
            bench(net=net_dgl, net_params=(g, features),
                  tag="DGL-primitives", nvprof=False, repeat=repeat, memory=True, log=log)

    run_baseline_and_graphiler(g, features)
    run_pyg(g, features)
    run_dgl(g, features)

    return log


if __name__ == '__main__':
    repeat = int(os.environ.get('REPEAT', 50))
    if len(sys.argv) != 3:
        print("usage: python GCN.py [dataset] [feat_dim]")
        exit()
    if sys.argv[1] == "all":
        log = {}
        for d in homo_dataset:
            log[d] = profile(d, homo_dataset[d], repeat)
        pd.DataFrame(log).to_pickle("./GCN.pkl")
    else:
        profile(sys.argv[1], int(sys.argv[2]), repeat)
