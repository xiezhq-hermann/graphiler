#include "optimizer.h"

#include <iostream>
#include <regex>

const static std::vector<std::string> BROADCAST = {
    "BroadcastSrcNode",     "BroadcastSrcNodeType", "BroadcastDstNode",
    "BroadcastDstNodeType", "BroadcastEdgeType",    "BroadcastNodeType"};

// Todo: more operators
const static std::vector<std::string> UNARY_DENSE = {
    "aten::relu", "aten::sigmoid", "aten::squeeze"};
// these two should be distinguished by input data residency instead of op name
const static std::vector<std::string> BINARY_DENSE = {"aten::mm", "aten::pow",
                                                      "aten::unsqueeze"};
const static std::vector<std::string> DUAL_DENSE = {"aten::mul"};
const static std::vector<std::string> TERNARY_DENSE = {};

const static std::vector<std::vector<std::string>> ORDINARY_DENSE = {
    UNARY_DENSE, BINARY_DENSE, TERNARY_DENSE};

namespace graphiler {
void reorder(std::shared_ptr<MPDFGAnnotation> &mpdfg) {

  // Todo: find a good way to incorporate residency into the pattern matcher
  // patterns for ordinary dense operators
  std::string scatter_dense = R"(
      graph(%ndata, %dglgraph __params__):
        %edata = my_ops::__broadcast__(%ndata, %dglgraph)
        %res = __op__(%edata __params__)
        return (%res, %edata))";

  // return edata as well to make it side effect free
  std::string dense_scatter = R"(
      graph(%ndata, %dglgraph __params__):
        %edata = my_ops::__broadcast__(%ndata, %dglgraph)
        %n_res = __op__(%ndata __params__)
        %res = my_ops::__broadcast__(%n_res, %dglgraph)
        return (%res, %edata))";

  // patterns for special binary dense operators
  std::string dual_scatter_dense = R"(
      graph(%ndata0, %ndata1, %dglgraph):
        %edata0 = my_ops::__broadcast__(%ndata0, %dglgraph)
        %edata1 = my_ops::__broadcast__(%ndata1, %dglgraph)
        %res = __op__(%edata0, %edata1)
        return (%res))";

  std::string dual_dense_scatter = R"(
      graph(%ndata0, %ndata1, %dglgraph):
        %n_res = __op__(%ndata0, %ndata1)
        %res = my_ops::__broadcast__(%n_res, %dglgraph)
        return (%res))";

  torch::jit::SubgraphRewriter rewriter;
  for (auto b : BROADCAST) {
    std::string pattern;
    std::string new_subgraph;
    for (auto d : DUAL_DENSE) {
      pattern = std::regex_replace(dual_scatter_dense,
                                   std::regex("__broadcast__"), b);
      pattern = std::regex_replace(pattern, std::regex("__op__"), d);
      new_subgraph = std::regex_replace(dual_dense_scatter,
                                        std::regex("__broadcast__"), b);
      new_subgraph = std::regex_replace(new_subgraph, std::regex("__op__"), d);
      rewriter.RegisterRewritePattern(pattern, new_subgraph);
      //   std::cout << pattern << new_subgraph;
    }
    std::string extra_params = "";
    for (size_t i = 0; i < ORDINARY_DENSE.size(); i++) {
      for (auto d : ORDINARY_DENSE[i]) {
        pattern = std::regex_replace(scatter_dense, std::regex(" __params__"),
                                     extra_params);
        pattern = std::regex_replace(pattern, std::regex("__broadcast__"), b);
        pattern = std::regex_replace(pattern, std::regex("__op__"), d);

        new_subgraph = std::regex_replace(
            dense_scatter, std::regex(" __params__"), extra_params);
        new_subgraph =
            std::regex_replace(new_subgraph, std::regex("__broadcast__"), b);
        new_subgraph =
            std::regex_replace(new_subgraph, std::regex("__op__"), d);
        rewriter.RegisterRewritePattern(pattern, new_subgraph);
        // std::cout << pattern << new_subgraph;
      }
      extra_params = extra_params + ", %param" + std::to_string(i);
    }

    // a special case which introduce unnecessary dependency to edge data
    // can get rid of it using reordering as well
    pattern = R"(
      graph(%ndata, %dglgraph):      
        %edata : Tensor = my_ops::__broadcast__(%ndata, %dglgraph)
        %feat_dim : int = my_ops::get_feat_dim(%edata)
        return (%feat_dim, %edata))";

    new_subgraph = R"(
      graph(%ndata, %dglgraph):
        %feat_dim : int = my_ops::get_feat_dim(%ndata)
        %edata : Tensor = my_ops::__broadcast__(%ndata, %dglgraph)
        return (%feat_dim, %edata))";
    rewriter.RegisterRewritePattern(
        std::regex_replace(pattern, std::regex("__broadcast__"), b),
        std::regex_replace(new_subgraph, std::regex("__broadcast__"), b));
  }

  rewriter.runOnGraph(mpdfg->DFG);
  dedup(mpdfg->DFG);
}
} // namespace graphiler