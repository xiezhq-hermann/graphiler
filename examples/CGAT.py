import sys

import torch
from torch import nn
import torch.nn.functional as F

from graphiler import EdgeBatchDummy, NodeBatchDummy, mpdfg_builder, update_all
from graphiler.utils import load_data, setup, check_equal, bench, homo_dataset, DEFAULT_DIM
device = setup()


# Todo: hyper message residency?
def message_func(edges: EdgeBatchDummy, fc_weight, attn_weight):
    z_s = torch.mm(edges.src['h'], fc_weight)
    z_d = torch.mm(edges.dst['h'], fc_weight)
    z2 = torch.cat([z_s, z_d], dim=1)
    a = torch.mm(z2, attn_weight)
    return {'z': z_s, 'e': torch.relu(a)}


def reduce_func(nodes: NodeBatchDummy):
    weight = nodes.mailbox['e']
    pos_weight = weight.unsqueeze(2)
    neg_weight = weight.unsqueeze(1)
    alpha = (neg_weight - pos_weight).clamp(min=0).sum(1)
    h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    return {"h": h}


mpdfg = mpdfg_builder(message_func, reduce_func)


class CGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CGATLayer, self).__init__()
        self.fc_weight = torch.rand(in_dim, out_dim).to(device)
        self.attn_weight = torch.rand(2 * out_dim, 1).to(device)

    def message_func(self, edges):
        z_s = torch.mm(edges.src['h'], self.fc_weight)
        z_d = torch.mm(edges.dst['h'], self.fc_weight)
        z2 = torch.cat([z_s, z_d], dim=1)
        a = torch.mm(z2, self.attn_weight)
        return {'z': z_s, 'e': F.leaky_relu(a)}

    def forward(self, g, feature, compile=False):
        g.ndata['h'] = feature
        if compile:
            update_all(g, mpdfg, msg_params=(
                self.fc_weight, self.attn_weight))
        else:
            g.update_all(self.message_func, reduce_func)
        return g.ndata.pop('h')


class CGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(CGAT, self).__init__()
        self.layer1 = CGATLayer(in_dim, hidden_dim)
        self.layer2 = CGATLayer(hidden_dim, out_dim)

    def forward(self, g, feature, compile=False):
        h = self.layer1(g, feature, compile=compile)
        h = F.elu(h)
        h = self.layer2(g, h, compile=compile)
        return h


def profile(dataset, feat_dim):
    g, features = load_data(dataset, feat_dim)
    g, features = g.to(device), features.to(device)

    net = CGAT(in_dim=features.size()[
               1], hidden_dim=DEFAULT_DIM, out_dim=DEFAULT_DIM).to(device)

    net.eval()
    with torch.no_grad():
        compile_res = bench(net=net, net_params=(
            g, features, True), tag="compile", nvprof=False)
        res = bench(net=net, net_params=(g, features, False),
                    tag="naive", nvprof=False)
        check_equal(compile_res, res)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python ConstrainedGAT.py [dataset] [feat_dim]")
        exit()
    if sys.argv[1] == "all":
        for d in homo_dataset:
            profile(d, homo_dataset[d])
    else:
        profile(sys.argv[1], int(sys.argv[2]))
