#include "builder.h"

namespace graphiler {
using namespace torch::jit;
void dfg_concat(std::shared_ptr<Graph> &mpdfg,
                const c10::intrusive_ptr<MPDFGAnnotation> &annotation,
                std::shared_ptr<Graph> &msg_graph,
                std::shared_ptr<Graph> &reduce_graph,
                at::optional<std::shared_ptr<Graph>> update_graph) {
  // Todo: control flow
  torch::jit::Inline(*msg_graph);
  auto msg_block = msg_graph->block();

  for (auto n : msg_block->nodes()) {
  }
}
} // namespace graphiler
