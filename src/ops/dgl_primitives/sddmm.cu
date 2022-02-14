/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/sddmm.cu
 * \brief SDDMM C APIs and definitions.
 */
#include "./functor.cuh"
#include "./sddmm.cuh"
#include "THC/THC.h"

namespace dgl {
namespace aten {

#define SWITCH_OP(op, Op, ...)                                                 \
  do {                                                                         \
    if ((op) == "add") {                                                       \
      typedef cuda::binary::Add<DType> Op;                                     \
      { __VA_ARGS__ }                                                          \
    } else if ((op) == "sub") {                                                \
      typedef cuda::binary::Sub<DType> Op;                                     \
      { __VA_ARGS__ }                                                          \
    } else if ((op) == "mul") {                                                \
      typedef cuda::binary::Mul<DType> Op;                                     \
      { __VA_ARGS__ }                                                          \
    } else if ((op) == "div") {                                                \
      typedef cuda::binary::Div<DType> Op;                                     \
      { __VA_ARGS__ }                                                          \
    } else if ((op) == "copy_lhs") {                                           \
      typedef cuda::binary::CopyLhs<DType> Op;                                 \
      { __VA_ARGS__ }                                                          \
    } else if ((op) == "copy_rhs") {                                           \
      typedef cuda::binary::CopyRhs<DType> Op;                                 \
      { __VA_ARGS__ }                                                          \
    } else if ((op) == "dot") {                                                \
      typedef cuda::binary::Dot<DType> Op;                                     \
      { __VA_ARGS__ }                                                          \
    } else {                                                                   \
      LOG(FATAL) << "Unsupported SpMM/SDDMM binary operator: " << op;          \
    }                                                                          \
  } while (0)

#define SWITCH_RHS(rhs_target, RhsTarget, ...)                                 \
  do {                                                                         \
    if ((rhs_target) == 0) {                                                   \
      constexpr int RhsTarget = 0;                                             \
      { __VA_ARGS__ }                                                          \
    } else if ((rhs_target) == 1) {                                            \
      constexpr int RhsTarget = 1;                                             \
      { __VA_ARGS__ }                                                          \
    } else if ((rhs_target) == 2) {                                            \
      constexpr int RhsTarget = 2;                                             \
      { __VA_ARGS__ }                                                          \
    } else {                                                                   \
      LOG(INFO) << "Invalid rhs target: " << (rhs_target);                     \
    }                                                                          \
  } while (0)

#define SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, ...)       \
  do {                                                                         \
    if ((lhs_target) == 0) {                                                   \
      constexpr int LhsTarget = 0;                                             \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                          \
    } else if ((lhs_target) == 1) {                                            \
      constexpr int LhsTarget = 1;                                             \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                          \
    } else if ((lhs_target) == 2) {                                            \
      constexpr int LhsTarget = 2;                                             \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                          \
    } else {                                                                   \
      LOG(INFO) << "Invalid lhs target: " << (lhs_target);                     \
    }                                                                          \
  } while (0)

/*!
 * \brief CUDA implementation of g-SDDMM on Coo format.
 */
template <typename IdType, int bits>
void SDDMMCoo(const std::string &op, int num_rows, int num_cols,
              at::Tensor coo_row, at::Tensor coo_col, at::Tensor lhs,
              at::Tensor rhs, at::Tensor out, int lhs_target, int rhs_target) {
  auto bcast = CalcBcastOff(op, lhs, rhs);

  SWITCH_BITS(bits, DType, {
    SWITCH_OP(op, Op, {
      SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
        cuda::SDDMMCoo<IdType, DType, Op, LhsTarget, RhsTarget>(
            bcast, num_rows, num_cols, coo_row, coo_col, lhs, rhs, out);
      });
    });
  });
}

template void SDDMMCoo<int32_t, 32>(const std::string &op, int num_rows,
                                    int num_cols, at::Tensor coo_row,
                                    at::Tensor coo_col, at::Tensor lhs,
                                    at::Tensor rhs, at::Tensor out,
                                    int lhs_target, int rhs_target);
template void SDDMMCoo<int64_t, 32>(const std::string &op, int num_rows,
                                    int num_cols, at::Tensor coo_row,
                                    at::Tensor coo_col, at::Tensor lhs,
                                    at::Tensor rhs, at::Tensor out,
                                    int lhs_target, int rhs_target);
template void SDDMMCoo<int32_t, 64>(const std::string &op, int num_rows,
                                    int num_cols, at::Tensor coo_row,
                                    at::Tensor coo_col, at::Tensor lhs,
                                    at::Tensor rhs, at::Tensor out,
                                    int lhs_target, int rhs_target);
template void SDDMMCoo<int64_t, 64>(const std::string &op, int num_rows,
                                    int num_cols, at::Tensor coo_row,
                                    at::Tensor coo_col, at::Tensor lhs,
                                    at::Tensor rhs, at::Tensor out,
                                    int lhs_target, int rhs_target);

} // namespace aten
} // namespace dgl

// Todo: alpha could be float if the feature is a float tensor
// by default it is is: int alpha = 1
torch::Tensor
SDDMMCoo_u_add_v_mul_alpha(torch::Tensor src_feature, torch::Tensor dst_feature,
                           int64_t alpha,
                           const c10::intrusive_ptr<DGLGraph> &graph) {
  int feat_dim = src_feature.dim() == 1 ? 1 : src_feature.size(1);
  auto out = torch::zeros({graph->num_edges, feat_dim},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
  dgl::aten::SDDMMCoo<int32_t, 32>("add", graph->num_nodes, graph->num_nodes,
                                   graph->COO_src, graph->COO_dst, src_feature,
                                   dst_feature, out, 0, 2);
  return out;
}

static auto registry = torch::RegisterOperators(
    "my_ops::gsddmm_u_add_v_mul_alpha(Tensor x, Tensor y, int alpha,"
    "__torch__.torch.classes.my_classes.DGLGraph g) -> Tensor z",
    &SDDMMCoo_u_add_v_mul_alpha);
