#include "optimizer.h"

#include <iostream>
#include <regex>

const static std::unordered_map<std::string, std::string> BROADCAST_TO_EDGE = {
    {"BroadcastSrcNode", "BroadcastSrcNodeType"},
    {"BroadcastDstNode", "BroadcastDstNodeType"}};

const static std::unordered_map<std::string, std::string> TYPE_BROADCAST_MM = {
    {"BroadcastNodeType", "SegmentMMNode"},
    {"BroadcastEdgeType", "SegmentMMEdge"}};

namespace graphiler {
void fusion(std::shared_ptr<MPDFGAnnotation> &mpdfg) {

  // Todo: implement more kernels and generalize the pattern
  // like the implementation of reorder
  // here we use specific operators because:
  // 1. already cover a large set of models
  // 2. only a small set of fused kernel are available

  // broadcast-reduce
  std::string scatter_reduce = R"(
      graph(%ndata, %dglgraph, %dims, %keep, %type):
        %edata = my_ops::BroadcastSrcNode(%ndata, %dglgraph)
        %res = my_ops::SpMMEdge(%edata, %dims, %keep, %type, %dglgraph)
        return (%res))";

  std::string gspmm_reduce = R"(
      graph(%ndata, %dglgraph, %dims, %keep, %type):
        %res = my_ops::SpMMSrc(%ndata, %dims, %keep, %type, %dglgraph)
        return (%res))";

  // template, gsddmm_u_op1_v_op2
  // broadcast-compute
  std::string u_add_v_mul_alpha = R"(
      graph(%udata, %vdata, %dglgraph, %alpha):
        %edata_src = my_ops::BroadcastSrcNode(%udata, %dglgraph)
        %edata_dst = my_ops::BroadcastDstNode(%vdata, %dglgraph)
        %res = aten::add(%edata_src, %edata_dst, %alpha)
        return (%res))";

  std::string gsddmm_u_add_v_mul_alpha = R"(
      graph(%udata, %vdata, %dglgraph, %alpha):
        %res = my_ops::gsddmm_u_add_v_mul_alpha(%udata, %vdata, %alpha, %dglgraph)
        return (%res))";

  // unsqueeze-squeeze
  std::string unsqueeze_squeeze = R"(
      graph(%data, %dim):
        %u_data : Tensor = aten::unsqueeze(%data, %dim)
        %res: Tensor = aten::squeeze(%u_data)
        return (%res))";

  std::string no_squeeze = R"(
      graph(%data, %dim):
        return (%data))";

  torch::jit::SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(scatter_reduce, gspmm_reduce);
  rewriter.RegisterRewritePattern(u_add_v_mul_alpha, gsddmm_u_add_v_mul_alpha);
  rewriter.RegisterRewritePattern(unsqueeze_squeeze, no_squeeze);

  // template, gspmm_u_op1_e_reduce
  // broadcast-compute-reduce
  std::string u_mul_e_reduce = R"(
      graph(%ndata, %msg, %dglgraph, %dims, %keep, %type):
        %edata = my_ops::__broadcast__(%ndata, %dglgraph)
        %mul = aten::mul(%msg, %edata)
        %res = my_ops::SpMMEdge(%mul, %dims, %keep, %type, %dglgraph)
        return (%res))";

  std::string gspmm_u_mul_e_reduce = R"(
      graph(%ndata, %msg, %dglgraph, %dims, %keep, %type):
        %res = my_ops::gspmm___source___mul_e_sum(%ndata, %msg, %dims, %keep, %type, %dglgraph)
        return (%res))";

  // template, ndata_ntype_scatter_op
  // broadcast-compute
  // Todo: it is also possible to implement it using reordering
  // "tensors must be 2-D" as indicated by TorchScript
  // therefore introduce squeeze and unsqueeze as a workaround
  std::string ndata_ntype_scatter_bmm = R"(
      graph(%ndata, %ntype_weight, %dglgraph):
        %edata : Tensor = my_ops::__broadcast_node__(%ndata, %dglgraph)
        %etype_weight : Tensor = my_ops::__broadcast_ntype__(%ntype_weight, %dglgraph)
        %res : Tensor = aten::bmm(%edata, %etype_weight)
        return (%res))";

  std::string ndata_ntype_segment_mm_scatter = R"(
      graph(%ndata, %ntype_weight, %dglgraph):
        %one : int = prim::Constant[value=1]()
        %s_ndata: Tensor = aten::squeeze(%ndata)
        %mm = my_ops::SegmentMMNode(%s_ndata, %ntype_weight, %dglgraph)
        %res : Tensor = my_ops::__broadcast_node__(%mm, %dglgraph)
        %u_res : Tensor = aten::unsqueeze(%res, %one)
        return (%u_res))";

  // template, type_scatter_op
  // broadcast compute
  std::string type_scatter_bmm = R"(
      graph(%edata, %etype_weight, %dglgraph):
        %edata_weight : Tensor = my_ops::__broadcast__(%etype_weight, %dglgraph)
        %res : Tensor = aten::bmm(%edata, %edata_weight)
        return (%res))";

  std::string type_segment_mm = R"(
      graph(%edata, %etype_weight, %dglgraph):
        %one : int = prim::Constant[value=1]()
        %s_edata: Tensor = aten::squeeze(%edata)
        %res = my_ops::__segment__(%s_edata, %etype_weight, %dglgraph)
        %u_res : Tensor = aten::unsqueeze(%res, %one)
        return (%u_res))";

  for (auto b : BROADCAST_TO_EDGE) {
    std::string pattern = std::regex_replace(
        u_mul_e_reduce, std::regex("__broadcast__"), b.first);
    std::string source =
        (b.first.find("Src") != std::string::npos) ? "src" : "dst";
    std::string new_subgraph = std::regex_replace(
        gspmm_u_mul_e_reduce, std::regex("__source__"), source);
    rewriter.RegisterRewritePattern(pattern, new_subgraph);

    pattern = std::regex_replace(ndata_ntype_scatter_bmm,
                                 std::regex("__broadcast_node__"), b.first);
    pattern = std::regex_replace(pattern, std::regex("__broadcast_ntype__"),
                                 b.second);
    new_subgraph =
        std::regex_replace(ndata_ntype_segment_mm_scatter,
                           std::regex("__broadcast_node__"), b.first);
    rewriter.RegisterRewritePattern(pattern, new_subgraph);
  }

  for (auto b : TYPE_BROADCAST_MM) {
    std::string pattern = std::regex_replace(
        type_scatter_bmm, std::regex("__broadcast__"), b.first);
    std::string new_subgraph = std::regex_replace(
        type_segment_mm, std::regex("__segment__"), b.second);
    rewriter.RegisterRewritePattern(pattern, new_subgraph);
  }

  rewriter.runOnGraph(mpdfg->DFG);
  dedup(mpdfg->DFG);
}
} // namespace graphiler