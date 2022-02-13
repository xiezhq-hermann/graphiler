import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all
from graphiler.utils import load_data, setup, check_equal, bench, hetero_dataset, DEFAULT_DIM

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


def profile(dataset, feat_dim):
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim)
    g, features = g.to(device), features.to(device)

    # graph format for PyG
    edge_type = g.edata['_TYPE']
    u, v = g.edges()
    adj = torch.stack([u, v]).to(device)

    # prepare heterogeneous graph
    g_hetero, _ = load_data(dataset, feat_dim, to_homo=False)
    g_hetero = g_hetero.to(device)
    rel_names = list(set(g_hetero.etypes))
    norm = torch.rand(g.num_edges(), 1).to(device)

    # instantiate nets for benchmark
    net = RGCN(feat_dim, DEFAULT_DIM,
               feat_dim, g.num_rels).to(device)
    net_dgl = RGCN_DGL(
        feat_dim, DEFAULT_DIM, feat_dim, g.num_rels).to(device)
    net_dgl_hetero = RGCN_DGL_hetero(
        feat_dim, DEFAULT_DIM, feat_dim, rel_names, len(rel_names)).to(device)
    net_pyg_bmm = RGCN_PyG(feat_dim, DEFAULT_DIM,
                           DEFAULT_DIM, g.num_rels, mode='bmm').to(device)
    net_pyg_slice = RGCN_PyG(feat_dim, DEFAULT_DIM,
                             DEFAULT_DIM, g.num_rels, mode='slice').to(device)

    net.eval()
    net_dgl.eval()
    net_dgl_hetero.eval()
    net_pyg_bmm.eval()
    net_pyg_slice.eval()
    with torch.no_grad():
        steps = 100
        bench(net=net_pyg_slice, net_params=(adj, features, edge_type),
              tag="PyG-slice", nvprof=False, steps=steps, memory=True)
        bench(net=net_dgl_hetero, net_params=(
            g_hetero, g_hetero.ndata['h']), tag="DGL-slice", nvprof=False, steps=steps, memory=True)
        compile_res = bench(net=net, net_params=(
            g, features, norm, True), tag="Graphiler", nvprof=False, steps=steps, memory=True)
        bench(net=net_pyg_bmm, net_params=(adj, features, edge_type),
              tag="PyG-bmm", nvprof=False, steps=steps, memory=True)
        bench(net=net_dgl, net_params=(
            g, features, g.edata['_TYPE'], norm), tag="DGL-bmm", nvprof=False, steps=steps, memory=True)
        res = bench(net=net, net_params=(g, features, norm, False),
                    tag="DGL-UDF", nvprof=False, steps=steps, memory=True)
        check_equal(compile_res, res)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage: python GCN.py [dataset] [feat_dim]")
        exit()
    if sys.argv[1] == "all":
        for d in hetero_dataset:
            profile(d, RGCN_FEAT_DIM)
    else:
        profile(sys.argv[1], int(sys.argv[2]))
