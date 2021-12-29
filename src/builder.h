#include "torch/csrc/jit/passes/inliner.h"

#include "../include/mpdfg.h"

namespace graphiler {
using torch::jit::Graph;
void DFG_concat(std::shared_ptr<MPDFGAnnotation> &mpdfg,
                std::shared_ptr<Graph> &msg_graph,
                std::shared_ptr<Graph> &reduce_graph,
                at::optional<std::shared_ptr<Graph>> update_graph);
} // namespace graphiler
