#include <torch/extension.h>

#include "builder/builder.h"
#include "optimizer/optimizer.h"

namespace graphiler {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("builder", &MPDFGBuilder, "build MPDFG");
  m.def("split", &split, "transform concat_mul to split_mul_sum");
  m.def("reorder", &reorder, "broadcast reordering");
  m.def("fusion", &fusion, "broadcast fusion");
  // using pybind because CustomClassHolder is managed by c10::intrusive_ptr
  // which is not compatible with torch::jit::Graph
  pybind11::class_<MPDFGAnnotation, std::shared_ptr<MPDFGAnnotation>>(
      m, "MPDFGAnnotation")
      .def(pybind11::init<std::shared_ptr<Graph>>());
}
} // namespace graphiler
