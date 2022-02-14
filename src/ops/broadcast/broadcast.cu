#include "../ops.cuh"

int64_t get_feat_dim(torch::Tensor features) {
  int64_t feat_dim = 1;
  for (int64_t i = 1; i < features.dim(); i++) {
    feat_dim *= features.size(i);
  }
  return feat_dim;
}

// Todo: COO version
inline torch::Tensor BroadcastBaseForward(torch::Tensor features, int num_nodes,
                                          int num_edges, torch::Tensor pointer,
                                          torch::Tensor indices) {
  CHECK_INPUT(features);

  int feat_dim = 1;
  int dims = features.dim();
  std::vector<int64_t> res_size = {num_edges};
  for (int i = 1; i < dims; i++) {
    feat_dim *= features.size(i);
    res_size.push_back(features.size(i));
  }

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
  result = torch::zeros(res_size,
                        torch::dtype(features.dtype()).device(torch::kCUDA));

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

torch::Tensor BroadcastSrcNode(torch::Tensor features,
                               const c10::intrusive_ptr<DGLGraph> &graph) {
  return BroadcastBaseForward(features, graph->num_nodes, graph->num_edges,
                              graph->out_pointer, graph->out_edge_indices);
}
torch::Tensor BroadcastDstNode(torch::Tensor features,
                               const c10::intrusive_ptr<DGLGraph> &graph) {
  return BroadcastBaseForward(features, graph->num_nodes, graph->num_edges,
                              graph->in_pointer, graph->in_edge_indices);
}

torch::Tensor BroadcastNodeType(torch::Tensor features,
                                const c10::intrusive_ptr<DGLGraph> &graph) {
  assert(graph->ntype_pointer.has_value());
  torch::Tensor node_indices = torch::arange(
      0, graph->num_nodes, torch::dtype(torch::kInt32).device(torch::kCUDA));
  return BroadcastBaseForward(features, graph->num_ntypes, graph->num_nodes,
                              graph->ntype_pointer.value(), node_indices);
}

torch::Tensor BroadcastSrcNodeType(torch::Tensor features,
                                   const c10::intrusive_ptr<DGLGraph> &graph) {
  return BroadcastSrcNode(BroadcastNodeType(features, graph), graph);
}

torch::Tensor BroadcastDstNodeType(torch::Tensor features,
                                   const c10::intrusive_ptr<DGLGraph> &graph) {
  return BroadcastDstNode(BroadcastNodeType(features, graph), graph);
}

torch::Tensor BroadcastEdgeType(torch::Tensor features,
                                const c10::intrusive_ptr<DGLGraph> &graph) {
  assert(graph->etype_pointer.has_value());
  torch::Tensor edge_indices = torch::arange(
      0, graph->num_edges, torch::dtype(torch::kInt32).device(torch::kCUDA));
  return BroadcastBaseForward(features, graph->num_etypes, graph->num_edges,
                              graph->etype_pointer.value(), edge_indices);
}

static auto registry =
    torch::RegisterOperators(
        "my_ops::BroadcastSrcNode(Tensor x, "
        "__torch__.torch.classes.my_classes.DGLGraph g) -> Tensor y",
        &BroadcastSrcNode)
        .op("my_ops::BroadcastDstNode(Tensor x, "
            "__torch__.torch.classes.my_classes.DGLGraph g) -> Tensor y",
            &BroadcastDstNode)
        .op("my_ops::BroadcastNodeType(Tensor x, "
            "__torch__.torch.classes.my_classes.DGLGraph g) -> Tensor y",
            &BroadcastNodeType)
        .op("my_ops::BroadcastSrcNodeType(Tensor x, "
            "__torch__.torch.classes.my_classes.DGLGraph g) -> Tensor y",
            &BroadcastSrcNodeType)
        .op("my_ops::BroadcastDstNodeType(Tensor x, "
            "__torch__.torch.classes.my_classes.DGLGraph g) -> Tensor y",
            &BroadcastDstNodeType)
        .op("my_ops::BroadcastEdgeType(Tensor x, "
            "__torch__.torch.classes.my_classes.DGLGraph g) -> Tensor y",
            &BroadcastEdgeType)
        .op("my_ops::get_feat_dim", &get_feat_dim);
