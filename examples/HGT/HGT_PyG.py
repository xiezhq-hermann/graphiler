import sys
import math

import dgl
import dgl.function as fn
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from graphiler.utils import setup
device = setup()


def masked_edge_index(edge_index, edge_mask):
    return edge_index[:, edge_mask]


def get_ntype(adj, edge_type, node_type, num_rels):
    adj = adj.cpu()
    edge_type = edge_type.cpu()
    node_type = node_type.cpu()
    src_ntype = [-1 for _ in range(num_rels)]
    dst_ntype = [-1 for _ in range(num_rels)]

    for i in range(len(edge_type)):
        u = adj[0, i].item()
        v = adj[1, i].item()
        etype = edge_type[i].item()
        src_ntype[etype] = node_type[u].item()
        dst_ntype[etype] = node_type[v].item()

    return src_ntype, dst_ntype


class HGTLayerSlice(MessagePassing):
    def __init__(self, in_feat_dim, out_feat_dim, num_node_types, num_rels):
        super(HGTLayerSlice, self).__init__(aggr='add')
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.sqrt_dk = math.sqrt(self.out_feat_dim)
        self.num_node_types = num_node_types
        self.num_rels = num_rels

        self.k_weights = torch.rand(
            self.num_node_types, self.in_feat_dim, self.out_feat_dim).to(device)
        self.q_weights = torch.rand(
            self.num_node_types, self.in_feat_dim, self.out_feat_dim).to(device)
        self.v_weights = torch.rand(
            self.num_node_types, self.in_feat_dim, self.out_feat_dim).to(device)
        self.a_weights = torch.rand(
            self.num_node_types, self.out_feat_dim, self.out_feat_dim).to(device)

        self.relation_pri = torch.ones(self.num_rels, 1).to(device)
        self.relation_att = torch.rand(
            self.num_rels, self.out_feat_dim, self.out_feat_dim).to(device)
        self.relation_msg = torch.rand(
            self.num_rels, self.out_feat_dim, self.out_feat_dim).to(device)

        self.skip = torch.ones(self.num_node_types).to(device)

    def upd(self, h, node_type):
        node_type = node_type.squeeze(-1)
        # (N, 1)
        skip = self.skip[node_type]
        # (N, out_dim, out_dim)
        a_weight = self.a_weights[node_type]
        # (N, 1)
        alpha = torch.sigmoid(skip)
        # (N, out_dim)
        trans_out = torch.bmm(h.unsqueeze(1), a_weight).squeeze()
        return trans_out * alpha.unsqueeze(1)

    def forward(self, h, adj, edge_type, node_type, src_type, dst_type):
        out = 0
        for i in range(self.num_rels):
            tmp = masked_edge_index(adj, edge_type == i)
            src_ntype = src_type[i]
            dst_ntype = dst_type[i]

            k = h @ self.k_weights[src_ntype]
            v = h @ self.v_weights[src_ntype]
            q = h @ self.q_weights[dst_ntype]
            k = k @ self.relation_att[i]
            v = v @ self.relation_msg[i]
            out_i = self.propagate(
                tmp, k=k, v=v, q=q, rel_pri=self.relation_pri[i])
            out = out + out_i

        out = self.upd(out, node_type)
        return out

    def message(self, k_j, v_j, q_i, edge_index_i, size_i, rel_pri):
        t = k_j * q_i
        attn_score = torch.sum(t, dim=1, keepdim=True) * rel_pri / self.sqrt_dk
        alpha = softmax(attn_score, edge_index_i, num_nodes=size_i)
        return v_j * alpha


class HGTLayer(MessagePassing):
    def __init__(self, in_feat_dim, out_feat_dim, num_node_types, num_rels):
        super(HGTLayer, self).__init__(aggr='add')
        # set the num_head to be 1
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.sqrt_dk = math.sqrt(self.out_feat_dim)
        self.num_node_types = num_node_types
        self.num_rels = num_rels

        self.k_weights = torch.rand(
            self.num_node_types, self.in_feat_dim, self.out_feat_dim).to(device)
        self.q_weights = torch.rand(
            self.num_node_types, self.in_feat_dim, self.out_feat_dim).to(device)
        self.v_weights = torch.rand(
            self.num_node_types, self.in_feat_dim, self.out_feat_dim).to(device)
        self.a_weights = torch.rand(
            self.num_node_types, self.out_feat_dim, self.out_feat_dim).to(device)

        self.relation_pri = torch.ones(self.num_rels, 1).to(device)
        self.relation_att = torch.rand(
            self.num_rels, self.out_feat_dim, self.out_feat_dim).to(device)
        self.relation_msg = torch.rand(
            self.num_rels, self.out_feat_dim, self.out_feat_dim).to(device)

        self.skip = torch.ones(self.num_node_types).to(device)

    def upd(self, h, node_type):
        node_type = node_type.squeeze(-1)
        # (N, 1)
        skip = self.skip[node_type]
        # (N, out_dim, out_dim)
        a_weight = self.a_weights[node_type]
        # (N, 1)
        alpha = torch.sigmoid(skip)
        # (N, out_dim)
        trans_out = torch.bmm(h.unsqueeze(1), a_weight).squeeze()
        return trans_out * alpha.unsqueeze(1)

    def forward(self, h, adj, edge_type, node_type, src_type, dst_type):
        node_type = node_type.unsqueeze(-1)
        h = self.propagate(adj, x=h, edge_type=edge_type, node_type=node_type)
        out = self.upd(h, node_type)
        return out

    def message(self, x_i, x_j, edge_index_i, edge_type, node_type_i, node_type_j, size_i):
        node_type_i = node_type_i.squeeze(-1)
        node_type_j = node_type_j.squeeze(-1)
        k_weight = self.k_weights[node_type_j]
        v_weight = self.v_weights[node_type_j]
        q_weight = self.q_weights[node_type_i]

        k = torch.bmm(x_j.unsqueeze(1), k_weight).squeeze()
        v = torch.bmm(x_j.unsqueeze(1), v_weight).squeeze()
        q = torch.bmm(x_i.unsqueeze(1), q_weight).squeeze()

        relation_att = self.relation_att[edge_type]
        relation_msg = self.relation_msg[edge_type]
        relation_pri = self.relation_pri[edge_type]

        k = torch.bmm(k.unsqueeze(1), relation_att).squeeze()
        v = torch.bmm(v.unsqueeze(1), relation_msg).squeeze()
        t = k * q
        attn_score = torch.sum(t, dim=1, keepdim=True) * \
            relation_pri / self.sqrt_dk
        alpha = softmax(attn_score, edge_index_i, num_nodes=size_i)

        rst = v * alpha
        return rst


class HGT_PyG(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_node_types, num_rels, mode='bmm'):
        super(HGT_PyG, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_node_types = num_node_types
        self.num_rels = num_rels

        Layer = HGTLayer if mode == 'bmm' else HGTLayerSlice

        self.layer0 = Layer(
            self.in_dim, self.h_dim, self.num_node_types, self.num_rels)
        self.layer1 = Layer(
            self.h_dim, self.out_dim, self.num_node_types, self.num_rels)

    def forward(self, adj, h, edge_type, node_type, src_type, dst_type):
        h = self.layer0(h, adj, edge_type, node_type, src_type, dst_type)
        h = self.layer1(h, adj, edge_type, node_type, src_type, dst_type)
        return h
