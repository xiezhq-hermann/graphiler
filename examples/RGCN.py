import sys

import dgl.function as fn
from dgl.nn.pytorch.conv import RelGraphConv
import dgl.nn.pytorch as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F

from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all
from graphiler.utils import load_data, setup, check_equal, bench, hetero_dataset, DEFAULT_DIM

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


class RGCN_DGL(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rels):
        super(RGCN_DGL, self).__init__()
        self.layer1 = RelGraphConv(in_dim, hidden_dim, num_rels)
        self.layer2 = RelGraphConv(hidden_dim, out_dim, num_rels)

    def forward(self, g, features, etypes, norm):
        x = F.relu(self.layer1(g, features, etypes, norm))
        x = self.layer2(g, x, etypes, norm)
        return x


class RelGraphConvHetero(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvHetero, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
             for rel in rel_names})

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(
                    len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(
                    self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(
                0)} for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(
                k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RGCN_DGL_hetero(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, rel_names, num_rels):
        super(RGCN_DGL_hetero, self).__init__()
        self.layer1 = RelGraphConvHetero(
            in_dim, hidden_dim, rel_names, num_rels)
        self.layer2 = RelGraphConvHetero(
            hidden_dim, out_dim, rel_names, num_rels)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = self.layer2(g, x)
        return x


def profile(dataset, feat_dim):
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim)
    g, features = g.to(device), features.to(device)

    net = RGCN(feat_dim, DEFAULT_DIM,
               feat_dim, g.num_rels).to(device)
    net_dgl = RGCN_DGL(
        feat_dim, DEFAULT_DIM, feat_dim, g.num_rels).to(device)

    g_hetero, _ = load_data(dataset, feat_dim, to_homo=False)
    g_hetero = g_hetero.to(device)
    rel_names = list(set(g_hetero.etypes))
    net_dgl_hetero = RGCN_DGL_hetero(features.size(
    )[1], DEFAULT_DIM, feat_dim, rel_names, len(rel_names)).to(device)

    norm = torch.rand(g.num_edges(), 1).to(device)

    net.eval()
    net_dgl.eval()
    with torch.no_grad():
        steps = 1000
        bench(net=net_dgl_hetero, net_params=(
            g_hetero, g_hetero.ndata['h']), tag="slice", nvprof=False, steps=steps, memory=True)
        compile_res = bench(net=net, net_params=(
            g, features, norm, True), tag="compile", nvprof=False, steps=steps, memory=True)
        bench(net=net_dgl, net_params=(
            g, features, g.edata['_TYPE'], norm), tag="graphconv", nvprof=False, steps=steps, memory=True)
        res = bench(net=net, net_params=(g, features, norm, False),
                    tag="naive", nvprof=False, steps=steps, memory=True)
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
