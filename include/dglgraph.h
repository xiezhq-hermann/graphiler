#ifndef INCLUDE_GRAPHILER_DGL_GRAPH_H_
#define INCLUDE_GRAPHILER_DGL_GRAPH_H_

#include <torch/custom_class.h>
#include <torch/script.h>

namespace graphiler {
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
        etype_pointer(etype_pointer){};

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
};
} // namespace graphiler
#endif // INCLUDE_GRAPHILER_DGL_GRAPH_H_
