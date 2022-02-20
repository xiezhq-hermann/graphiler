import numpy as np
from pathlib import Path

import torch

from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset

import dgl
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset

DEFAULT_DIM = 64
DGL_PATH = str(Path.home()) + "/.dgl/"
torch.classes.load_library(DGL_PATH + "libgraphiler.so")

homo_dataset = {"cora": 1433, "pubmed": 500,
                "ppi": 50, "arxiv": 128, "reddit": 602}

hetero_dataset = ["aifb", "mutag", "bgs", "biokg", "am"]


def load_data(name, feat_dim=DEFAULT_DIM, prepare=True, to_homo=True):
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
    elif name == "wikikg2":
        dataset = DglLinkPropPredDataset(name='ogbl-wikikg2')
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

    if name in hetero_dataset:
        g, type_pointers = prepare_hetero_graph_simplified(g, node_feats)
        if to_homo:
            g = dgl.to_homogeneous(g)
            if prepare:
                g = prepare_graph(g)
                g.DGLGraph.SetNtypePointer(
                    type_pointers['ntype_node_pointer'].cuda())
                g.DGLGraph.SetEtypePointer(
                    type_pointers['etype_edge_pointer'].cuda())
                g.DGLGraph.SetNtypeCOO(
                    g.ndata['_TYPE'].type(torch.LongTensor).cuda())
                g.DGLGraph.SetEtypeCOO(
                    g.edata['_TYPE'].type(torch.LongTensor).cuda())

            g.num_ntypes = len(type_pointers['ntype_node_pointer']) - 1
            # note #rels is different to #etypes in some cases
            # for simplicity we use these two terms interchangeably
            # and refer an edge type as (src_type, etype, dst_type)
            # see DGL document for more information
            g.num_rels = num_etypes = len(
                type_pointers['etype_edge_pointer']) - 1
    elif prepare:
        g = prepare_graph(g)
    g.ntype_data = {}
    g.etype_data = {}
    return g, node_feats


def prepare_graph(g, ntype=None):
    # Todo: integrate with dgl.graph, long int, multiple devices

    reduce_node_index = g.in_edges(g.nodes(ntype))[0]
    reduce_node_index = reduce_node_index.type(torch.IntTensor).cuda()
    reduce_edge_index = g.in_edges(
        g.nodes(ntype), form='eid').type(torch.IntTensor).cuda()
    message_node_index = (g.out_edges(g.nodes(ntype))[
        1]).type(torch.IntTensor).cuda()
    message_edge_index = g.out_edges(
        g.nodes(ntype), form='eid').type(torch.IntTensor).cuda()
    assert(len(reduce_node_index) == len(reduce_edge_index) == len(
        message_node_index) == len(message_edge_index) == g.num_edges())

    src, dst = g.edges()
    Coosrc, Coodst = src.type(
        torch.LongTensor).cuda(), dst.type(torch.LongTensor).cuda()

    reduce_node_pointer = [0] + g.in_degrees(g.nodes(ntype)).tolist()
    message_node_pointer = [0] + g.out_degrees(g.nodes(ntype)).tolist()

    for i in range(1, len(reduce_node_pointer)):
        reduce_node_pointer[i] += reduce_node_pointer[i - 1]
        message_node_pointer[i] += message_node_pointer[i - 1]
    reduce_node_pointer = torch.IntTensor(reduce_node_pointer).cuda()
    message_node_pointer = torch.IntTensor(message_node_pointer).cuda()

    g.DGLGraph = torch.classes.my_classes.DGLGraph(
        reduce_node_pointer, reduce_node_index, reduce_edge_index, message_node_pointer, message_node_index, message_edge_index, Coosrc, Coodst, None, None)

    return g


def prepare_hetero_graph_simplified(g, features, nkey='h'):
    ntype_id = {name: i for i, name in enumerate(g.ntypes)}
    ntype_pointer = np.cumsum(
        [0] + [g.number_of_nodes(ntype) for ntype in g.ntypes])
    for ntype, i in ntype_id.items():
        g.nodes[ntype].data[nkey] = features[ntype_pointer[i]:ntype_pointer[i + 1]]

    etype_pointer = [0]
    for etype in g.canonical_etypes:
        g_sub = g[etype]
        etype_pointer.append(etype_pointer[-1] + g_sub.num_edges())

    return (g, {"ntype_node_pointer": torch.IntTensor(ntype_pointer), "etype_edge_pointer": torch.IntTensor(etype_pointer)})


def setup(device='cuda:0'):
    torch.manual_seed(42)
    assert torch.cuda.is_available()
    device = torch.device(device)
    return device


if __name__ == "__main__":
    # a place for testing data loading
    for dataset in homo_dataset:
        load_data(dataset)
    for dataset in hetero_dataset:
        load_data(dataset)
