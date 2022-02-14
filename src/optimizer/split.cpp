#include "optimizer.h"

namespace graphiler {

void split(std::shared_ptr<MPDFGAnnotation> &mpdfg) {
  // specific pattern for GAT and its variants
  torch::jit::Inline(*mpdfg->DFG);
  // Todo: explicitly using residency to assist pattern matching
  std::string concat_mul = R"(
      graph(%src, %dst, %weight, %dim):
        %list : Tensor[] = prim::ListConstruct(%src, %dst)
        %concat : Tensor = aten::cat(%list, %dim)
        %res : Tensor = aten::mm(%concat, %weight)
        return (%res))";

  std::string split_mul_sum = R"(
      graph(%src, %dst, %weight, %dim):
        %zero : int = prim::Constant[value=0]()
        %one : int = prim::Constant[value=1]()
        %dim_s : int = my_ops::get_feat_dim(%src)
        %dim_d : int = my_ops::get_feat_dim(%dst)
        %split_size : int[] = prim::ListConstruct(%dim_s, %dim_d)
        %splited : Tensor[] = aten::split(%weight, %split_size, %zero)
        %weight_src : Tensor, %weight_dst : Tensor = prim::ListUnpack(%splited)
        %res_src : Tensor = aten::mm(%src, %weight_src)
        %res_dst : Tensor = aten::mm(%dst, %weight_dst)
        %res : Tensor = aten::add(%res_src, %res_dst, %one)
        return (%res))";

  torch::jit::SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(concat_mul, split_mul_sum);
  rewriter.runOnGraph(mpdfg->DFG);
  dedup(mpdfg->DFG);
}
} // namespace graphiler