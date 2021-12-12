#include "builder.h"
#include <torch/extension.h>

namespace graphiler {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("builder", &dfg_concat, "concat");
}
} // namespace graphiler
