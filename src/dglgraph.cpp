#include "../include/dglgraph.h"

namespace graphiler {

TORCH_LIBRARY(my_classes, m) {
  m.class_<DGLGraph>("DGLGraph")
      // Currently, registering overloaded constructors are not supported
      // .def(torch::init<>())
      .def(
          torch::init<torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, at::optional<torch::Tensor>,
                      at::optional<torch::Tensor>>())
      .def("SetNtypePointer", &DGLGraph::SetNtypePointer)
      .def("SetEtypePointer", &DGLGraph::SetEtypePointer)
      // not a good idea to wrap primitives as member functions
      // .def("BroadcastSrcNode", &DGLGraph::BroadcastSrcNodeForward)
      // .def("BroadcastDstNode", &DGLGraph::BroadcastDstNodeForward)
      // .def("SegmentSoftmax", &DGLGraph::SegmentSoftmaxForward)
      // .def("SpMM", &DGLGraph::SpMMCsrForward)
      ;
}
} // namespace graphiler
