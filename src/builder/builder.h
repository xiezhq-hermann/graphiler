#include <algorithm>
#include <iostream>
#include <unordered_set>

#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>

#include "../../include/mpdfg.h"

namespace graphiler {
using torch::jit::Graph;
void DFG_concat(std::shared_ptr<MPDFGAnnotation> &mpdfg,
                std::shared_ptr<Graph> &msg_graph,
                std::shared_ptr<Graph> &reduce_graph,
                at::optional<std::shared_ptr<Graph>> update_graph);
} // namespace graphiler
