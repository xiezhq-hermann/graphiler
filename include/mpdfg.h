#ifndef INCLUDE_GRAPHILER_MPDFG_H_
#define INCLUDE_GRAPHILER_MPDFG_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include <unordered_map>

namespace graphiler {
enum Residency { Node, Edge, NodeType, EdgeType, Shared };
enum Movement { Broadcast, BroadcastSrc, BroadcastDst, Reduce, Norm, Dense };
enum Stage { Creation, Aggregation, Update };

struct MPDFGAnnotation {
  std::shared_ptr<torch::jit::Graph> DFG;
  std::unordered_map<size_t, Residency> data_residency;
  std::unordered_map<torch::jit::Node *, Movement> data_movement;
  MPDFGAnnotation(){};
  MPDFGAnnotation(std::shared_ptr<torch::jit::Graph> DFG) : DFG(DFG){};
};
} // namespace graphiler
#endif // INCLUDE_GRAPHILER_MPDFG_H_