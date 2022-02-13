#include <unordered_set>

#include "torch/csrc/jit/ir/subgraph_matcher.h"
#include "torch/csrc/jit/passes/subgraph_rewrite.h"
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>

#include "../../include/mpdfg.h"

namespace graphiler {
void dedup(std::shared_ptr<torch::jit::Graph> &graph);
void split(std::shared_ptr<MPDFGAnnotation> &mpdfg);
} // namespace graphiler
