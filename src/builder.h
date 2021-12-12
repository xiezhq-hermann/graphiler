#include "torch/csrc/jit/passes/inliner.h"

#include "../include/mpdfg.h"

namespace graphiler {
using namespace torch::jit;
void dfg_concat(std::shared_ptr<Graph> &mpdfg,
                const c10::intrusive_ptr<MPDFGAnnotation> &annotation,
                std::shared_ptr<Graph> &msg_graph,
                std::shared_ptr<Graph> &reduce_graph,
                at::optional<std::shared_ptr<Graph>> update_graph);
} // namespace graphiler
