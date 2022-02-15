import sys
import os
from timeit import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all
from graphiler.utils import load_data, setup, check_equal, bench, hetero_dataset, DEFAULT_DIM, init_log, empty_cache

from RGCN_DGL import RGCN_DGL, RGCN_DGL_hetero
from RGCN_PyG import RGCN_PyG

device = setup()

# to successfully benchmark this model using Seastar, a smaller feature dimension is used
RGCN_FEAT_DIM = 16


def message_func(edges: EdgeBatchDummy):
    relation_weight = edges.type['weight']
    msg = torch.bmm(edges.src['h'].unsqueeze(1), relation_weight).squeeze()
    msg = msg * edges.data['norm']
    return {'m': msg}


def reduce_func(nodes: NodeBatchDummy):
    return {'h': torch.sum(nodes.mailbox['m'], dim=1)}


mpdfg = mpdfg_builder(message_func, reduce_func)


class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels):
        super(RGCNLayer, self).__init__()
        self.weight = torch.rand(num_rels, in_dim, out_dim).to(device)

    def message_func(self, edges):
        relation_weight = self.weight[edges.data['_TYPE']]
        msg = torch.bmm(edges.src['h'].unsqueeze(1), relation_weight).squeeze()
        msg = msg * edges.data['norm']
        return {'m': msg}

    def forward(self, g, feature, norm, compile=False):
        g.ndata['h'] = feature
        g.edata['norm'] = norm
        g.etype_data['weight'] = self.weight
        if compile:
            update_all(g, mpdfg)
        else:
            g.update_all(self.message_func, reduce_func)
            # use built-in as the DGL-bmm baseline
            # g.update_all(self.message_func, fn.sum('m', 'h'))
        return g.ndata.pop('h')


class RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels):
        super(RGCN, self).__init__()

        self.layer1 = RGCNLayer(in_dim, hidden_dim, num_rels)
        self.layer2 = RGCNLayer(hidden_dim, out_dim, num_rels)

    def forward(self, g, features, norm, compile=False):
        x = self.layer1(g, features, norm, compile)
        x = F.relu(x)
        x = self.layer2(g, x, norm, compile)
        return x


def profile(dataset, feat_dim, repeat=1000):
    log = init_log(["PyG-slice", "DGL-slice",
                    "Graphiler", "PyG-bmm", "DGL-bmm", "DGL-UDF"], ["time", "mem"])
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim)
    g_hetero, _ = load_data(dataset, feat_dim, to_homo=False)
    features = features.to(device)

    @empty_cache
    def run_baseline_graphiler(g, features):
        g = g.to(device)
        norm = torch.rand(g.num_edges(), 1).to(device)
        net = RGCN(feat_dim, DEFAULT_DIM,
                   feat_dim, g.num_rels).to(device)
        net.eval()
        with torch.no_grad():
            compile_res = bench(net=net, net_params=(
                g, features, norm, True), tag="Graphiler", nvprof=False, repeat=repeat, memory=True, log=log)
            res = bench(net=net, net_params=(g, features, norm, False),
                        tag="DGL-UDF", nvprof=False, repeat=repeat, memory=True, log=log)
            check_equal(compile_res, res)
        del g, norm, net, compile_res, res

    @empty_cache
    def run_dgl_hetero(g_hetero, features):
        g_hetero = g_hetero.to(device)
        rel_names = list(set(g_hetero.etypes))
        net_dgl_hetero = RGCN_DGL_hetero(
            feat_dim, DEFAULT_DIM, feat_dim, rel_names, len(rel_names)).to(device)
        net_dgl_hetero.eval()
        with torch.no_grad():
            bench(net=net_dgl_hetero, net_params=(
                g_hetero, g_hetero.ndata['h']), tag="DGL-slice", nvprof=False, repeat=repeat, memory=True, log=log)
        del g_hetero, rel_names, net_dgl_hetero

    @empty_cache
    def run_pyg_bmm(g, features):
        edge_type = g.edata['_TYPE']
        u, v = g.edges()
        adj = torch.stack([u, v]).to(device)
        net_pyg_bmm = RGCN_PyG(feat_dim, DEFAULT_DIM,
                               DEFAULT_DIM, g.num_rels, mode='bmm').to(device)
        net_pyg_bmm.eval()
        with torch.no_grad():
            bench(net=net_pyg_bmm, net_params=(adj, features, edge_type),
                  tag="PyG-bmm", nvprof=False, repeat=repeat, memory=True, log=log)
        del edge_type, u, v, adj, net_pyg_bmm

    @empty_cache
    def run_dgl_bmm(g, features):
        g = g.to(device)
        norm = torch.rand(g.num_edges(), 1).to(device)
        net_dgl = RGCN_DGL(
            feat_dim, DEFAULT_DIM, feat_dim, g.num_rels).to(device)
        net_dgl.eval()
        with torch.no_grad():
            bench(net=net_dgl, net_params=(
                g, features, g.edata['_TYPE'], norm), tag="DGL-bmm", nvprof=False, repeat=repeat, memory=True, log=log)
        del g, norm, net_dgl

    @empty_cache
    def run_pyg_slice(g, features):
        edge_type = g.edata['_TYPE']
        u, v = g.edges()
        adj = torch.stack([u, v]).to(device)
        net_pyg_slice = RGCN_PyG(feat_dim, DEFAULT_DIM,
                                 DEFAULT_DIM, g.num_rels, mode='slice').to(device)
        net_pyg_slice.eval()
        with torch.no_grad():
            bench(net=net_pyg_slice, net_params=(adj, features, edge_type),
                  tag="PyG-slice", nvprof=False, repeat=repeat, memory=True, log=log)
        del edge_type, u, v, adj, net_pyg_slice

    run_baseline_graphiler(g, features)
    run_dgl_bmm(g, features)
    run_dgl_hetero(g_hetero, features)
    run_pyg_bmm(g, features)
    run_pyg_slice(g, features)

    return log


if __name__ == '__main__':
    repeat = int(os.environ.get('REPEAT', 50))
    if len(sys.argv) != 3:
        print("usage: python GCN.py [dataset] [feat_dim]")
        exit()
    if sys.argv[1] == "all":
        log = {}
        for d in hetero_dataset:
            log[d] = profile(d, RGCN_FEAT_DIM, repeat)
        pd.DataFrame(log).to_pickle("./RGCN.pkl")
    else:
        profile(sys.argv[1], int(sys.argv[2]), repeat)
