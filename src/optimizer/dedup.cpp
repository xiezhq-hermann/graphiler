#include "optimizer.h"

namespace graphiler {

void dedup(std::shared_ptr<torch::jit::Graph> &graph) {

  EliminateDeadCode(graph);
  ConstantPooling(graph);
  EliminateCommonSubexpression(graph);

  // same value used by multiple users
  // potential side effect?
  std::unordered_set<torch::jit::Node *> nodes_to_delete;
  std::unordered_set<torch::jit::Node *, torch::jit::HashNode,
                     torch::jit::EqualNode>
      unique_nodes;
  for (auto n : graph->block()->nodes()) {
    auto match = unique_nodes.find(n);
    if (match != unique_nodes.end()) {
      n->replaceAllUsesWith(*match);
      nodes_to_delete.insert(n);
    } else {
      unique_nodes.insert(n);
    }
  }
  for (auto n : nodes_to_delete) {
    n->removeAllInputs();
  }
  for (auto n : nodes_to_delete) {
    n->destroy();
  }
  unique_nodes.clear();
  nodes_to_delete.clear();
}
} // namespace graphiler