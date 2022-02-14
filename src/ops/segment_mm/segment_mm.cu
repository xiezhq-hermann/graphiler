#include "../ops.cuh"

// Todo: faster implementation using native cuda
inline torch::Tensor SegmentMMBase(torch::Tensor features,
                                   torch::Tensor weights,
                                   torch::Tensor type_pointer, int num_rels) {
  CHECK_INPUT(features);
  CHECK_INPUT(weights);

  type_pointer = type_pointer.to(torch::kCPU);

  std::vector<torch::Tensor> results;
  auto type_pointer_acc = type_pointer.accessor<int, 1>();
  for (int i = 0; i < num_rels; i++) {
    torch::Tensor input = features.index(
        {torch::indexing::Slice(type_pointer_acc[i], type_pointer_acc[i + 1])});
    torch::Tensor weight = weights.index({i});
    results.push_back(at::mm(input, weight));
  }
  return torch::cat(results, 0);
}

torch::Tensor SegmentMMNode(torch::Tensor features, torch::Tensor weights,
                            const c10::intrusive_ptr<DGLGraph> &graph) {
  return SegmentMMBase(features, weights, graph->ntype_pointer.value(),
                       graph->num_ntypes);
}

torch::Tensor SegmentMMEdge(torch::Tensor features, torch::Tensor weights,
                            const c10::intrusive_ptr<DGLGraph> &graph) {
  return SegmentMMBase(features, weights, graph->etype_pointer.value(),
                       graph->num_etypes);
}

static auto registry =
    torch::RegisterOperators(
        "my_ops::SegmentMMNode(Tensor x, Tensor y, "
        "__torch__.torch.classes.my_classes.DGLGraph g) -> Tensor z",
        &SegmentMMNode)
        .op("my_ops::SegmentMMEdge(Tensor x, Tensor y, "
            "__torch__.torch.classes.my_classes.DGLGraph g) -> Tensor z",
            &SegmentMMEdge);
