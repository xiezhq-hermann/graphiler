/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/spmm.cu
 * \brief SPMM C APIs and definitions.
 */
#include "cuda_common.h"
#include "functor.cuh"
#include "spmm.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <cusparse_v2.h>
#include <torch/torch.h>

namespace dgl {

using namespace cuda;

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
    } else {                                                                   \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op;                \
    }                                                                          \
  } while (0)

/*!
 * \brief CUDA implementation of g-SpMM on Csr format.
 * \note use cusparse if the reduce operator is `sum` and there is
 *       no broadcast, use dgl's kernel in other cases.
 */
template <typename IdType, int bits>
void SpMMCsr(const std::string &op, const std::string &reduce, int64_t num_rows,
             int64_t num_cols, at::Tensor csr_indptr, at::Tensor csr_indices,
             at::Tensor csr_data, at::Tensor ufeat, at::Tensor efeat,
             at::Tensor out, std::vector<at::Tensor> out_aux) {
  auto bcast = CalcBcastOff(op, ufeat, efeat);
  if (reduce == "sum") {
    if (op == "copy_lhs") { // no edge data.
      SWITCH_BITS(bits, DType, {
        int feat_len = 1;
        auto ufeat_shp = ufeat.sizes();
        for (int i = 1; i < ufeat.ndimension(); ++i) {
          feat_len *= ufeat_shp[i];
        }
        int m = num_rows, n = feat_len, k = num_cols;
        int nnz = csr_indices.size(0);
        DType alpha = 1., beta = 0.;
        cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
        auto edge_weight = torch::ones(
            {
                nnz,
            },
            ufeat.options());

        cusparseSpMatDescr_t matA;
        cusparseDnMatDescr_t matB, matC;

        constexpr auto dtype = dgl::runtime::cuda_dtype<DType>::value;
        constexpr auto idtype = dgl::runtime::cusparse_idtype<IdType>::value;
        CUSPARSE_CALL(cusparseCreateCsr(
            &matA, m, k, nnz, csr_indptr.data_ptr<IdType>(),
            csr_indices.data_ptr<IdType>(), edge_weight.data_ptr<DType>(),
            idtype, idtype, CUSPARSE_INDEX_BASE_ZERO, dtype));
        CUSPARSE_CALL(cusparseCreateDnMat(&matB, k, n, n,
                                          ufeat.data_ptr<DType>(), dtype,
                                          CUSPARSE_ORDER_ROW));
        CUSPARSE_CALL(cusparseCreateDnMat(&matC, m, n, n, out.data_ptr<DType>(),
                                          dtype, CUSPARSE_ORDER_ROW));

        auto transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        auto transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        size_t workspace_size;
        CUSPARSE_CALL(cusparseSpMM_bufferSize(
            handle, transA, transB, &alpha, matA, matB, &beta, matC, dtype,
            CUSPARSE_SPMM_CSR_ALG2, &workspace_size));
        void *workspace;
        cudaMalloc(&workspace, workspace_size);
        CUSPARSE_CALL(cusparseSpMM(handle, transA, transB, &alpha, matA, matB,
                                   &beta, matC, dtype, CUSPARSE_SPMM_CSR_ALG2,
                                   workspace));
        cudaFree(workspace);
        CUSPARSE_CALL(cusparseDestroySpMat(matA));
        CUSPARSE_CALL(cusparseDestroyDnMat(matB));
        CUSPARSE_CALL(cusparseDestroyDnMat(matC));
      });
    } else {
      SWITCH_BITS(bits, DType, {
        SWITCH_OP(op, Op, {
          cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Sum<IdType, DType>>(
              bcast, num_rows, num_cols, csr_indptr, csr_indices, csr_data,
              ufeat, efeat, out, out_aux[0], out_aux[1]);
        });
      });
    }
  } else if (reduce == "max") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Max<IdType, DType>>(
            bcast, num_rows, num_cols, csr_indptr, csr_indices, csr_data, ufeat,
            efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else if (reduce == "min") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Min<IdType, DType>>(
            bcast, num_rows, num_cols, csr_indptr, csr_indices, csr_data, ufeat,
            efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

template void SpMMCsr<int32_t, 32>(const std::string &op,
                                   const std::string &reduce, int64_t num_rows,
                                   int64_t num_cols, at::Tensor csr_indptr,
                                   at::Tensor csr_indices, at::Tensor csr_data,
                                   at::Tensor ufeat, at::Tensor efeat,
                                   at::Tensor out,
                                   std::vector<at::Tensor> out_aux);
template void SpMMCsr<int64_t, 32>(const std::string &op,
                                   const std::string &reduce, int64_t num_rows,
                                   int64_t num_cols, at::Tensor csr_indptr,
                                   at::Tensor csr_indices, at::Tensor csr_data,
                                   at::Tensor ufeat, at::Tensor efeat,
                                   at::Tensor out,
                                   std::vector<at::Tensor> out_aux);
template void SpMMCsr<int32_t, 64>(const std::string &op,
                                   const std::string &reduce, int64_t num_rows,
                                   int64_t num_cols, at::Tensor csr_indptr,
                                   at::Tensor csr_indices, at::Tensor csr_data,
                                   at::Tensor ufeat, at::Tensor efeat,
                                   at::Tensor out,
                                   std::vector<at::Tensor> out_aux);
template void SpMMCsr<int64_t, 64>(const std::string &op,
                                   const std::string &reduce, int64_t num_rows,
                                   int64_t num_cols, at::Tensor csr_indptr,
                                   at::Tensor csr_indices, at::Tensor csr_data,
                                   at::Tensor ufeat, at::Tensor efeat,
                                   at::Tensor out,
                                   std::vector<at::Tensor> out_aux);

torch::Tensor
SpMMCsr_u_mul_e_sum(torch::Tensor ufeature, torch::Tensor efeature,
                    std::vector<int64_t> dims, bool keep_dim,
                    at::optional<at::ScalarType> dtype, torch::Tensor node_id,
                    torch::Tensor node_index, torch::Tensor edge_index,
                    torch::Tensor node_pointer) {
  CHECK_INPUT(ufeature);
  CHECK_INPUT(efeature);
  CHECK_INPUT(node_id);
  CHECK_INPUT(node_index);
  CHECK_INPUT(edge_index);
  CHECK_INPUT(node_pointer);

  int num_nodes = node_id.sizes()[0];
  auto out =
      torch::zeros({num_nodes, max(ufeature.sizes()[1], efeature.sizes()[1])},
                   torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto empty = at::empty({0}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  std::vector<at::Tensor> empty_list;
  empty_list.push_back(empty);
  empty_list.push_back(empty);
  SpMMCsr<int32_t, 32>("mul", "sum", num_nodes, num_nodes, node_pointer,
                       node_index, edge_index, ufeature, efeature, out,
                       empty_list);
  return out;
}

// edata is (E, 1)
__global__ void u_mul_e_sum_kernel(const int num_nodes, const int feat_dim,
                                   const float *ufeature, const float *efeature,
                                   const int *node_index, const int *edge_index,
                                   const int *node_pointer,
                                   float *__restrict__ next_layer) {
  int node_id = blockDim.y * blockIdx.x + threadIdx.y;
  if (node_id >= num_nodes)
    return;
  int offset = node_pointer[node_id];
  int end = node_pointer[node_id + 1];
  float local = 0.0f;
  int target;
  for (int i = offset; i < end; i++) {
    const int eid = __ldg(edge_index + i);
    const int cid = __ldg(node_index + i);
    target = cid * feat_dim + threadIdx.x;
    local += ufeature[target] * efeature[eid];
  }
  next_layer[node_id * feat_dim + threadIdx.x] = local;
}

/*
  Every warp process a tile of tile_r (# of nodes) * tile_c (# of features)
  blockDim.x = 32
  tile_r = blockDim.y
  tile_c * gridDim.y >= feat_dim
  tile_c = 32 * factor
*/
template <int tile_r, int tile_c, int factor>
__global__ void u_mul_e_sum_kernel_neat(
    const int num_nodes, const int feat_dim, const float *ufeature,
    const float *efeature, const int *node_index, const int *edge_index,
    const int *node_pointer, float *__restrict__ next_layer) {
  int node_id = tile_r * blockIdx.x + threadIdx.y;
  if (node_id >= num_nodes)
    return;
  int offset = node_pointer[node_id];
  int degree = node_pointer[node_id + 1] - offset;
  int sm_offset = threadIdx.y << 5;
  __shared__ int neighbor_local[tile_r << 5];
  __shared__ float factor_local[tile_r << 5];
  float local[factor];
#pragma unroll
  for (int i = 0; i < factor; i++) {
    local[i] = 0;
  }

  // Tree reduction might be useful for certain graphs
  int feat_id = blockIdx.y * tile_c + threadIdx.x;
  for (int i = 0; i < degree / 32; i++) {
    neighbor_local[sm_offset + threadIdx.x] =
        node_index[offset + i * 32 + threadIdx.x] * feat_dim;
    factor_local[sm_offset + threadIdx.x] =
        efeature[edge_index[offset + i * 32 + threadIdx.x]];
#pragma unroll
    for (int j = 0; j < 32; j++) {
      // const int eid = __ldg(edge_index + offset + i * 32 + j);
      // float local_factor = efeature[eid];
      int local_target = neighbor_local[sm_offset + j] + feat_id;
      float local_factor = factor_local[sm_offset + j];
#pragma unroll
      for (int k = 0; k < factor; k++) {
        local[k] += ufeature[local_target + (k << 5)] * local_factor;
      }
    }
  }

  if (threadIdx.x < degree % 32) {
    neighbor_local[sm_offset + threadIdx.x] =
        node_index[offset + degree - (degree % 32) + threadIdx.x] * feat_dim;
    factor_local[sm_offset + threadIdx.x] =
        efeature[edge_index[offset + degree - (degree % 32) + threadIdx.x]];
  }
  __syncwarp();
  for (int i = 0; i < degree % 32; i++) {
    float local_factor = factor_local[sm_offset + i];
    // const int eid = __ldg(edge_index + offset + degree - (degree % 32) + i);
    // float local_factor = efeature[eid];
    int local_target = neighbor_local[sm_offset + i] + feat_id;
#pragma unroll
    for (int k = 0; k < factor; k++) {
      local[k] += ufeature[local_target + (k << 5)] * local_factor;
    }
  }
#pragma unroll
  for (int i = 0; i < factor; i++) {
    next_layer[node_id * feat_dim + feat_id + i * 32] = local[i];
  }
}

// ad hoc implementation
torch::Tensor u_mul_e_sum(torch::Tensor ufeature, torch::Tensor efeature,
                          std::vector<int64_t> dims, bool keep_dim,
                          at::optional<at::ScalarType> dtype,
                          torch::Tensor node_id, torch::Tensor node_index,
                          torch::Tensor edge_index,
                          torch::Tensor node_pointer) {

  const int num_nodes = node_id.sizes()[0];
  const int feat_dim = ufeature.size(1);
  auto out = torch::zeros({num_nodes, feat_dim},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
  // Todo: the template parameters should be changed for different GPU models
  if (feat_dim % 64 == 0) {
    // dim3 blocks((num_nodes + 7) / 8, feat_dim / 64, 1);
    // dim3 threads(32, 8, 1);
    // u_mul_e_sum_kernel_neat<8, 64, 2><<<blocks, threads>>>(
    dim3 blocks(num_nodes, 1, 1);
    dim3 threads(32, 1, 1);
    u_mul_e_sum_kernel_neat<1, 64, 2><<<blocks, threads>>>(
        num_nodes, feat_dim, ufeature.data_ptr<float>(),
        efeature.data_ptr<float>(), node_index.data_ptr<int>(),
        edge_index.data_ptr<int>(), node_pointer.data_ptr<int>(),
        out.data_ptr<float>());
  } else if (feat_dim % 32 == 0) {
    dim3 blocks((num_nodes + 3) / 4, 1, 1);
    dim3 threads(32, 4, 1);
    u_mul_e_sum_kernel_neat<4, 32, 1><<<blocks, threads>>>(
        num_nodes, feat_dim, ufeature.data_ptr<float>(),
        efeature.data_ptr<float>(), node_index.data_ptr<int>(),
        edge_index.data_ptr<int>(), node_pointer.data_ptr<int>(),
        out.data_ptr<float>());
  } else {
    dim3 blocks((num_nodes + 15) / 16, 1, 1);
    dim3 threads(feat_dim, 16, 1);
    u_mul_e_sum_kernel<<<blocks, threads>>>(
        num_nodes, feat_dim, ufeature.data_ptr<float>(),
        efeature.data_ptr<float>(), node_index.data_ptr<int>(),
        edge_index.data_ptr<int>(), node_pointer.data_ptr<int>(),
        out.data_ptr<float>());
  }
  return out;
}

} // namespace aten
} // namespace dgl

// Todo: simplify the interfaces
torch::Tensor SpMMCsrForward(torch::Tensor features, std::vector<int64_t> dims,
                             bool keep_dim, at::optional<at::ScalarType> dtype,
                             const c10::intrusive_ptr<DGLGraph> &graph) {
  int feat_dim = features.dim() == 1 ? 1 : features.size(1);
  auto out = torch::zeros({graph->num_nodes, feat_dim},
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto empty =
      at::empty({0}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  std::vector<at::Tensor> empty_list;
  empty_list.push_back(empty);
  // Todo: ad hoc implementation, should be separated
  if (features.sizes()[0] == graph->num_nodes) {
    dgl::aten::SpMMCsr<int32_t, 32>("copy_lhs", "sum", graph->num_nodes,
                                    graph->num_nodes, graph->in_pointer,
                                    graph->in_node_indices, empty, features,
                                    empty, out, empty_list);
  } else {
    dgl::aten::SpMMCsr<int32_t, 32>("copy_lhs", "sum", graph->num_nodes,
                                    graph->num_nodes, graph->in_pointer,
                                    graph->in_edge_indices, empty, features,
                                    empty, out, empty_list);
  }
  return out;
}

static auto registry = torch::RegisterOperators(
    "my_ops::SpMM(Tensor x, int[] dim, bool k, int? t, "
    "__torch__.torch.classes.my_classes.DGLGraph g) -> Tensor y",
    &SpMMCsrForward);
// Todo
// .op("my_ops::gspmm_u_mul_e_sum", &u_mul_e_sum);
