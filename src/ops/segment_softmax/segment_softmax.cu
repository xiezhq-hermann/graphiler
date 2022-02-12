#include "../ops.cuh"

torch::Tensor SegmentSoftmaxForward(torch::Tensor features, int64_t dim,
                                    at::optional<at::ScalarType> dtype,
                                    const c10::intrusive_ptr<DGLGraph> &graph) {

  CHECK_INPUT(features);
  assert(dim == 1);

  torch::Tensor result = torch::zeros_like(features);

  int feat_dim = features.dim() == 1 ? 1 : features.size(1);
  if (feat_dim == 1) {
    dim3 threads(32, 1, 1);
    dim3 blocks(graph->num_nodes, 1, 1);
    softmax_v<<<blocks, threads>>>(
        features.data_ptr<float>(), graph->in_pointer.data_ptr<int>(),
        graph->in_edge_indices.data_ptr<int>(), result.data_ptr<float>());
  } else {
    assert(false && "kernel to be implemented");
    // Todo: softmax for multi head attention
  }
  return result;
}

// enum class ScalarType : int8_t
static auto registry = torch::RegisterOperators(
    "my_ops::SegmentSoftmax(Tensor x, int dim, int? t, "
    "__torch__.torch.classes.my_classes.DGLGraph g) -> Tensor y",
    &SegmentSoftmaxForward);
