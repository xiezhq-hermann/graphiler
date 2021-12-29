#include "../include/dglgraph.h"

namespace graphiler {

TORCH_LIBRARY(my_classes, m) {
  m.class_<DGLGraph>("DGLGraph")
      .def(
          torch::init<torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, at::optional<torch::Tensor>,
                      at::optional<torch::Tensor>>());
}
} // namespace graphiler
