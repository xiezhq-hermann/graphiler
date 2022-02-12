#include "../ops.cuh"

inline torch::Tensor BroadcastBase(torch::Tensor features, int num_nodes,
                                   int num_edges, torch::Tensor pointer,
                                   torch::Tensor indices) {
  CHECK_INPUT(features);

  int dims = features.dim();
  int feat_dim = dims == 1 ? 1 : features.size(1);

  // configure block sizes based on
  // column-wise and instruction-level parallelism
  int factor = 1;
  int num_threads = feat_dim;
  while (num_threads > 1024) {
    num_threads = (num_threads + 1) / 2;
    factor <<= 1;
  }
  int row_per_block = (128 + num_threads - 1) / num_threads;
  dim3 blocks((num_nodes + row_per_block - 1) / row_per_block, factor, 1);
  dim3 threads(num_threads, row_per_block, 1);

  torch::Tensor result;
  if (dims == 2) {
    result = torch::zeros({num_edges, feat_dim},
                          torch::dtype(features.dtype()).device(torch::kCUDA));
  } else if (dims == 1) {
    result = torch::zeros({num_edges},
                          torch::dtype(features.dtype()).device(torch::kCUDA));
  } else {
    assert(false && "feature size not supported");
  }

  // feature data can be various types
  if (features.dtype() == torch::kFloat32) {
    scatter<float>
        <<<blocks, threads>>>(num_nodes, feat_dim, features.data_ptr<float>(),
                              pointer.data_ptr<int>(), indices.data_ptr<int>(),
                              result.data_ptr<float>());
  } else if (features.dtype() == torch::kInt64) {
    scatter<int64_t>
        <<<blocks, threads>>>(num_nodes, feat_dim, features.data_ptr<int64_t>(),
                              pointer.data_ptr<int>(), indices.data_ptr<int>(),
                              result.data_ptr<int64_t>());
  } else {
    assert(false && "feature type not supported");
  }

  return result;
}

torch::Tensor
BroadcastSrcNodeForward(torch::Tensor features,
                        const c10::intrusive_ptr<DGLGraph> &graph) {
  return BroadcastBase(features, graph->num_nodes, graph->num_edges,
                       graph->out_pointer, graph->out_edge_indices);
}
torch::Tensor
BroadcastDstNodeForward(torch::Tensor features,
                        const c10::intrusive_ptr<DGLGraph> &graph) {
  return BroadcastBase(features, graph->num_nodes, graph->num_edges,
                       graph->in_pointer, graph->in_edge_indices);
}

// Todo: ntype and etype data broadcast
// torch::Tensor DGLGraph::BroadcastDstNodeForward(torch::Tensor features) {
//   return BroadcastBase(features, num_nodes, num_edges, in_pointer,
//                        in_edge_indices);
// }
// torch::Tensor BroadcastNodeType(torch::Tensor features, graphiler::DGLGraph
// graph) {
//   return BroadcastBase(features, graph.num_ntypes, graph.ntype_pointer,
//                 graph.node_indices);
// }
// torch::Tensor BroadcastEdgeType(torch::Tensor features, graphiler::DGLGraph
// graph) {
//   return BroadcastBase(features, graph.num_etypes, graph.etype_pointer,
//                 graph.edge_indices);
// }

// torch::Tensor BroadcastSrcNodeType(torch::Tensor features,
// graphiler::DGLGraph graph); torch::Tensor BroadcastDstNodeType(torch::Tensor
// features, graphiler::DGLGraph graph);

static auto registry =
    torch::RegisterOperators(
        "my_ops::BroadcastSrcNode(Tensor x, "
        "__torch__.torch.classes.my_classes.DGLGraph g) -> Tensor y",
        &BroadcastSrcNodeForward)
        .op("my_ops::BroadcastDstNode(Tensor x, "
            "__torch__.torch.classes.my_classes.DGLGraph g) -> Tensor y",
            &BroadcastDstNodeForward);
