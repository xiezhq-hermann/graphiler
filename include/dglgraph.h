#ifndef INCLUDE_GRAPHILER_DGL_GRAPH_H_
#define INCLUDE_GRAPHILER_DGL_GRAPH_H_

#include <torch/custom_class.h>
#include <torch/script.h>

struct DGLGraph : torch::CustomClassHolder {
  // minimal graph object
  // Todo: integrate with dgl.graph
  DGLGraph(){};
  DGLGraph(torch::Tensor in_pointer, torch::Tensor in_node_indices,
           torch::Tensor in_edge_indices, torch::Tensor out_pointer,
           torch::Tensor out_node_indices, torch::Tensor out_edge_indices,
           torch::Tensor COO_src, torch::Tensor COO_dst,
           at::optional<torch::Tensor> ntype_pointer,
           at::optional<torch::Tensor> etype_pointer)
      : in_pointer(in_pointer), in_node_indices(in_node_indices),
        in_edge_indices(in_edge_indices), out_pointer(out_pointer),
        out_node_indices(out_node_indices), out_edge_indices(out_edge_indices),
        COO_src(COO_src), COO_dst(COO_dst), ntype_pointer(ntype_pointer),
        etype_pointer(etype_pointer) {
    num_nodes = in_pointer.size(0) - 1;
    num_edges = in_node_indices.size(0);
    num_ntypes =
        ntype_pointer.has_value() ? ntype_pointer.value().size(0) - 1 : 1;
    num_etypes =
        etype_pointer.has_value() ? etype_pointer.value().size(0) - 1 : 1;
  };
  inline void SetNtypePointer(torch::Tensor pointer) {
    ntype_pointer = pointer;
    num_ntypes =
        ntype_pointer.has_value() ? ntype_pointer.value().size(0) - 1 : 1;
  }
  inline void SetEtypePointer(torch::Tensor pointer) {
    etype_pointer = pointer;
    num_etypes =
        etype_pointer.has_value() ? etype_pointer.value().size(0) - 1 : 1;
  }
  inline void SetNtypeCOO(torch::Tensor pointer) {
    ntype_COO = pointer;
    assert(pointer.size(0) == num_nodes);
  }
  inline void SetEtypeCOO(torch::Tensor pointer) {
    etype_COO = pointer;
    assert(pointer.size(0) == num_edges);
  }

  // Todo: int64 support
  int num_nodes;
  int num_edges;
  int num_ntypes;
  int num_etypes;

  // for message reduce, or CSC format
  torch::Tensor in_pointer;
  torch::Tensor in_node_indices;
  torch::Tensor in_edge_indices;

  // for message creation, or CSR format
  torch::Tensor out_pointer;
  torch::Tensor out_node_indices;
  torch::Tensor out_edge_indices;

  // for certain operators
  torch::Tensor COO_src;
  torch::Tensor COO_dst;

  // for heterogeneous graph
  // ids of nodes/edges with the same type are assumed to be continuous
  at::optional<torch::Tensor> ntype_pointer;
  at::optional<torch::Tensor> etype_pointer;

  at::optional<torch::Tensor> ntype_COO;
  at::optional<torch::Tensor> etype_COO;
};
#endif // INCLUDE_GRAPHILER_DGL_GRAPH_H_
