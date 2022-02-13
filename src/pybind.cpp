#include <torch/extension.h>

#include "builder/builder.h"

namespace graphiler {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("builder", &DFG_concat, "concat");
  // using pybind because CustomClassHolder is managed by c10::intrusive_ptr
  // which is not compatible with torch::jit::Graph
  pybind11::class_<MPDFGAnnotation, std::shared_ptr<MPDFGAnnotation>>(
      m, "MPDFGAnnotation")
      .def(pybind11::init<std::shared_ptr<Graph>>());
}
} // namespace graphiler
