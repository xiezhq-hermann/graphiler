#ifndef INCLUDE_GRAPHILER_MPDFG_H_
#define INCLUDE_GRAPHILER_MPDFG_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include <unordered_map>

namespace graphiler {
// using torch::jit::Graph;
enum Residency { Node, Edge, NodeType, EdgeType, Shared };
enum Movement { Broadcast, BroadcastSrc, BroadcastDst, Reduce, Norm, Dense };
enum Stage { Creation, Aggregation, Update };

struct MPDFGAnnotation : torch::CustomClassHolder {
  // torch::jit::Graph is not compatible with CustomClassHolder
  // because the object is managed by c10::intrusive_ptr
  // std::shared_ptr<Graph> DFG;
  // void dfg_concat(std::shared_ptr<Graph> &mpdfg,
  //                 std::shared_ptr<Graph> &msg_graph,
  //                 std::shared_ptr<Graph> &reduce_graph,
  //                 at::optional<std::shared_ptr<Graph>> update_graph);
  std::unordered_map<int, Residency> data_residency;
  std::unordered_map<int, Movement> data_movement;
  std::unordered_map<int, Stage> mp_stage;
  MPDFGAnnotation(){};
};
} // namespace graphiler
#endif // INCLUDE_GRAPHILER_MPDFG_H_