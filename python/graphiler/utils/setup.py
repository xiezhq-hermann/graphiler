import math
import numpy as np

import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
import torch

DEFAULT_DIM = 64

homo_dataset = {"cora": 1433, "pubmed": 500,
                "ppi": 50, "arxiv": 128, "reddit": 602}

hetero_dataset = ["debug_hetero", "aifb", "mutag",
                  "bgs", "biokg", "am", "wikikg", "mag"]


def load_data(name, feat_dim=DEFAULT_DIM, prepare=True):
    if name in homo_dataset:
        feat_dim = homo_dataset[name]
    if name == "arxiv":
        dataset = DglNodePropPredDataset(name="ogbn-arxiv")
        g = dataset[0][0]
    elif name == "proteins":
        dataset = DglNodePropPredDataset(name="ogbn-proteins")
        g = dataset[0][0]
    elif name == "reddit":
        dataset = dgl.data.RedditDataset()
        g = dataset[0]
    elif name == "ppi":
        g = dgl.batch([g for x in ["train", "test", "valid"]
                       for g in dgl.data.PPIDataset(x)])
    elif name == "cora":
        dataset = dgl.data.CoraGraphDataset()
        g = dataset[0]
    elif name == "pubmed":
        dataset = dgl.data.PubmedGraphDataset()
        g = dataset[0]
    elif name == "debug":
        g = dgl.graph(([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]))
    elif name == 'aifb':
        dataset = AIFBDataset()
        g = dataset[0]
    elif name == 'mutag':
        dataset = MUTAGDataset()
        g = dataset[0]
    elif name == 'bgs':
        dataset = BGSDataset()
        g = dataset[0]
    elif name == 'am':
        dataset = AMDataset()
        g = dataset[0]
    elif name == "mag":
        dataset = DglNodePropPredDataset(name="ogbn-mag")
        g = dataset[0][0]
    elif name == "wikikg":
        dataset = DglLinkPropPredDataset(name='ogbl-wikikg')
        g = dataset[0]
        src, dst = g.edges()
        reltype = torch.flatten(g.edata['reltype']).cuda()
        num_etypes = torch.max(reltype).item() + 1
        hetero_dict = {}
        for i in range(num_etypes):
            type_index = (reltype == i).nonzero()
            hetero_dict[('n', str(i), 'n')] = (
                torch.flatten(src[type_index]), torch.flatten(dst[type_index]))
        g = dgl.heterograph(hetero_dict)
    elif name == "biokg":
        dataset = DglLinkPropPredDataset(name='ogbl-biokg')
        g = dataset[0]
    elif name == 'debug_hetero':
        g = dgl.heterograph({
            ('user', '+1', 'movie'): ([0, 0, 1], [0, 1, 0]),
            ('user', '-1', 'movie'): ([1, 2, 2], [1, 0, 1]),
            ('user', '+1', 'user'): ([0], [1]),
            ('user', '-1', 'user'): ([2], [1]),
            ('movie', '+1', 'movie'): ([0], [1]),
            ('movie', '-1', 'movie'): ([1], [0])
        })
    else:
        raise Exception("Unknown Dataset")

    node_feats = torch.rand([g.number_of_nodes(), feat_dim])

    if prepare:
        if name in hetero_dataset:
            type_pointers = prepare_hetero_graph_simplified(g)
            g = prepare_graph(dgl.to_homogeneous(g))
            g.ntype_pointer = type_pointers['ntype_node_pointer']
            g.etype_pointer = type_pointers['etype_edge_pointer']
            g.num_ntypes = max(g.ndata[dgl.NTYPE]).item() + 1
            g.num_etypes = max(g.edata[dgl.ETYPE]).item() + 1
        else:
            g = prepare_graph(g)
    return g, node_feats


def prepare_graph(g, ntype=None):
    # Todo: integrate with dgl.graph
    # Todo: long int, multiple devices
    g.node_id = g.nodes(ntype).type(torch.IntTensor).cuda()

    g.reduce_node_index = (g.in_edges(g.nodes(ntype))[
                           0]).type(torch.IntTensor).cuda()
    g.reduce_edge_index = g.in_edges(
        g.nodes(ntype), form='eid').type(torch.IntTensor).cuda()
    g.message_node_index = (g.out_edges(g.nodes(ntype))[
                            1]).type(torch.IntTensor).cuda()
    g.message_edge_index = g.out_edges(
        g.nodes(ntype), form='eid').type(torch.IntTensor).cuda()
    assert(len(g.reduce_node_index) == len(g.reduce_edge_index) == len(
        g.message_node_index) == len(g.message_edge_index) == g.num_edges())

    src, dst = g.edges()
    g.Coosrc, g.Coodst = src.type(
        torch.IntTensor).cuda(), dst.type(torch.IntTensor).cuda()

    reduce_node_pointer = [0] + g.in_degrees(g.nodes(ntype)).tolist()
    message_node_pointer = [0] + g.out_degrees(g.nodes(ntype)).tolist()

    for i in range(1, len(reduce_node_pointer)):
        reduce_node_pointer[i] += reduce_node_pointer[i - 1]
        message_node_pointer[i] += message_node_pointer[i - 1]
    g.reduce_node_pointer = torch.IntTensor(reduce_node_pointer).cuda()
    g.message_node_pointer = torch.IntTensor(message_node_pointer).cuda()

    return g


def prepare_hetero_graph_simplified(g):
    ntype_pointer = np.cumsum(
        [0] + [g.number_of_nodes(ntype) for ntype in g.ntypes])

    etype_pointer = [0]
    for etype in g.canonical_etypes:
        g_sub = g[etype]
        etype_pointer.append(etype_pointer[-1] + g_sub.num_edges())

    return{"ntype_node_pointer": torch.IntTensor(ntype_pointer).cuda(), "etype_edge_pointer": torch.IntTensor(etype_pointer).cuda()}


def setup(device='cuda:0'):
    torch.manual_seed(42)
    assert torch.cuda.is_available()
    device = torch.device(device)
    return device


if __name__ == "__main__":
    # a place for testing data loading
    pass
