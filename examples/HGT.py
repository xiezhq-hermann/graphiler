import sys
import math

import torch
import torch.nn as nn

import dgl.function as fn
from dgl.nn.functional import edge_softmax

from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all
from graphiler.utils import load_data, setup, check_equal, bench, hetero_dataset, DEFAULT_DIM

device = setup()


# Todo: explanation on the interface difference
def message_func(edges: EdgeBatchDummy, sqrt_dk: float):

    k_weight = edges.srctype['k_weight']
    v_weight = edges.srctype['v_weight']
    q_weight = edges.dsttype['q_weight']

    k = torch.bmm(edges.src['h'].unsqueeze(1), k_weight).squeeze()
    v = torch.bmm(edges.src['h'].unsqueeze(1), v_weight).squeeze()
    q = torch.bmm(edges.dst['h'].unsqueeze(1), q_weight).squeeze()

    relation_att = edges.type['relation_att']
    relation_msg = edges.type['relation_msg']
    relation_pri = edges.type['relation_pri']

    k = torch.bmm(k.unsqueeze(1), relation_att).squeeze()
    v = torch.bmm(v.unsqueeze(1), relation_msg).squeeze()
    t = k * q
    attn_score = torch.sum(t, dim=1, keepdim=True) * relation_pri / sqrt_dk
    return {'attn': attn_score, 'v': v}


def reduce_func(nodes: NodeBatchDummy):
    t = torch.softmax(nodes.mailbox['attn'], dim=1)
    m = t * nodes.mailbox['v']
    return {'t': torch.sum(m, dim=1)}


def update_func(nodes: NodeBatchDummy):
    skip = nodes.type['skip']
    a_weight = nodes.type['a_weight']
    alpha = torch.sigmoid(skip)
    trans_out = torch.bmm(nodes.data['t'].unsqueeze(1), a_weight).squeeze()
    return {'h': trans_out * alpha.unsqueeze(1)}


mpdfg = mpdfg_builder(message_func, reduce_func, update_func)


class HGTLayer_simplified(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, num_ntypes, num_rels):
        super(HGTLayer_simplified, self).__init__()
        # set the num_head to be 1
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.sqrt_dk = math.sqrt(self.out_feat_dim)
        self.num_ntypes = num_ntypes
        self.num_rels = num_rels

        self.k_weight = torch.rand(
            self.num_ntypes, self.in_feat_dim, self.out_feat_dim).to(device)
        self.q_weight = torch.rand(
            self.num_ntypes, self.in_feat_dim, self.out_feat_dim).to(device)
        self.v_weight = torch.rand(
            self.num_ntypes, self.in_feat_dim, self.out_feat_dim).to(device)
        self.a_weight = torch.rand(
            self.num_ntypes, self.out_feat_dim, self.out_feat_dim).to(device)

        self.relation_pri = torch.ones(self.num_rels, 1).to(device)
        self.relation_att = torch.rand(
            self.num_rels, self.out_feat_dim, self.out_feat_dim).to(device)
        self.relation_msg = torch.rand(
            self.num_rels, self.out_feat_dim, self.out_feat_dim).to(device)

        self.skip = torch.ones(self.num_ntypes).to(device)

    def message_func(self, edges):
        # (E, in_dim, out_dim)
        k_weight = self.k_weight[edges.src['_TYPE']]
        v_weight = self.v_weight[edges.src['_TYPE']]
        q_weight = self.q_weight[edges.dst['_TYPE']]
        # (E, 1)
        relation_pri = self.relation_pri[edges.data['_TYPE']]
        # (E, out_dim, out_dim)
        relation_att = self.relation_att[edges.data['_TYPE']]
        # (E, out_dim, out_dim)
        relation_msg = self.relation_msg[edges.data['_TYPE']]

        # (E, out_dim) <- (E, 1, in_dim) * (E, in_dim, out_dim)
        k = torch.bmm(edges.src['h'].unsqueeze(1), k_weight).squeeze()
        v = torch.bmm(edges.src['h'].unsqueeze(1), v_weight).squeeze()
        q = torch.bmm(edges.dst['h'].unsqueeze(1), q_weight).squeeze()

        # (E, out_dim) <- (E, 1, out_dim) * (E, out_dim, out_dim)
        k = torch.bmm(k.unsqueeze(1), relation_att).squeeze()
        v = torch.bmm(v.unsqueeze(1), relation_msg).squeeze()
        # (E, out_dim)
        t = k * q
        # (E, 1)
        attn_score = torch.sum(t, dim=1, keepdims=True) * \
            relation_pri / self.sqrt_dk
        return {'attn': attn_score, 'v': v}

    # def msg_func_softmax(self, edges):
    #     self.message_func(edges)

    def update_func(self, nodes):
        # (N, 1)
        skip = self.skip[nodes.data['_TYPE']]
        # (N, out_dim, out_dim)
        a_weight = self.a_weight[nodes.data['_TYPE']]
        # (N, 1)
        alpha = torch.sigmoid(skip)
        # (N, out_dim)
        trans_out = torch.bmm(nodes.data['t'].unsqueeze(1), a_weight).squeeze()
        return {'h': trans_out * alpha.unsqueeze(1)}

    def forward(self, g, h, compile=False):
        g.ndata['h'] = h
        g.ntype_data['k_weight'] = self.k_weight
        g.ntype_data['v_weight'] = self.v_weight
        g.ntype_data['q_weight'] = self.q_weight
        g.etype_data['relation_pri'] = self.relation_pri
        g.etype_data['relation_att'] = self.relation_att
        g.etype_data['relation_msg'] = self.relation_msg
        g.ntype_data['skip'] = self.skip
        g.ntype_data['a_weight'] = self.a_weight

        if compile:
            update_all(g, mpdfg, msg_params=(self.sqrt_dk,))
        else:
            g.update_all(self.message_func, reduce_func, self.update_func)
            # use dgl built-in functions as dgl-batch baseline
            # g.apply_edges(self.message_func)
            # g.edata['m'] = edge_softmax(g, g.edata['attn']) * g.edata['v']
            # g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 't'), self.update_func)


class HGT(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_ntypes, num_rels):
        super(HGT, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_ntypes = num_ntypes
        self.num_rels = num_rels

        self.layer0 = HGTLayer_simplified(
            self.in_dim, self.h_dim, self.num_ntypes, self.num_rels)
        self.layer1 = HGTLayer_simplified(
            self.h_dim, self.out_dim, self.num_ntypes, self.num_rels)

    def forward(self, g, h, compile=False):
        self.layer0(g, h, compile=compile)
        self.layer1(g, g.ndata['h'], compile=compile)
        return g.ndata.pop('h')


class HGTLayerHetero(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads=1,
                 dropout=0.2,
                 use_norm=False):
        super(HGTLayerHetero, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(
            torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(
            self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(
            self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[(srctype, etype, dsttype)]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop(
                    't').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype: (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't'))
                                for etype, e_id in edge_dict.items()}, cross_reducer='mean')

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha  # + h[ntype] * (1-alpha) ?
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class NetHetero(nn.Module):
    def __init__(self, node_dict, edge_dict, in_dim, h_dim, out_dim):
        super(NetHetero, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.layer0 = HGTLayerHetero(in_dim, h_dim, node_dict, edge_dict)
        self.layer1 = HGTLayerHetero(h_dim, out_dim, node_dict, edge_dict)

    def forward(self, G, h):
        h = self.layer0(G, h)
        h = self.layer1(G, h)


def profile(dataset, feat_dim):
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim)
    g, features = g.to(device), features.to(device)

    g_hetero, _ = load_data(dataset, feat_dim, to_homo=False)
    g_hetero = g_hetero.to(device)
    node_dict = {}
    edge_dict = {}
    for ntype in g_hetero.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in g_hetero.canonical_etypes:
        edge_dict[etype] = len(edge_dict)

    net = HGT(features.size()[1], DEFAULT_DIM,
              DEFAULT_DIM, g.num_ntypes, g.num_rels).to(device)
    net_hetero = NetHetero(node_dict, edge_dict,
                           features.size()[1], DEFAULT_DIM, DEFAULT_DIM).to(device)

    net.eval()
    net_hetero.eval()
    with torch.no_grad():
        steps = 1000
        bench(net=net_hetero, net_params=(g_hetero, g_hetero.ndata['h']),
              tag="HGT_slice on {}".format(dataset), nvprof=False, steps=steps, memory=True)
        compile_res = bench(net=net, net_params=(
            g, features, True), tag="compile on {}".format(dataset), nvprof=False, steps=steps, memory=True)
        res = bench(net=net, net_params=(g, features, False),
                    tag="naive", nvprof=False, steps=steps, memory=True)
        check_equal(compile_res, res)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage: python HGT.py [dataset] [feat_dim]")
        exit()
    if sys.argv[1] == "all":
        for d in hetero_dataset:
            profile(d, DEFAULT_DIM)
    else:
        profile(sys.argv[1], int(sys.argv[2]))
