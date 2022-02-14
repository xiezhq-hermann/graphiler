#include "optimizer.h"

namespace graphiler {

void dedup(std::shared_ptr<torch::jit::Graph> &graph) {

  // provided by TorchScript but not working for introduced my_ops
  EliminateDeadCode(graph);
  ConstantPooling(graph);
  EliminateCommonSubexpression(graph);

  // same value used by multiple users
  // values are not used by any users
  std::unordered_set<torch::jit::Node *> nodes_to_delete;
  std::unordered_set<torch::jit::Node *, torch::jit::HashNode,
                     torch::jit::EqualNode>
      unique_nodes;
  for (auto n : graph->block()->nodes()) {
    std::string kind = n->kind().toQualString();
    if (kind.find("my_ops::") != std::string::npos) {
      if (n->hasUses()) {
        auto match = unique_nodes.find(n);
        if (match != unique_nodes.end()) {
          n->replaceAllUsesWith(*match);
          nodes_to_delete.insert(n);
        } else {
          unique_nodes.insert(n);
        }
      } else {
        nodes_to_delete.insert(n);
      }
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