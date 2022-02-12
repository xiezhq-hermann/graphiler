/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/spmm.cuh
 * \brief SPMM CUDA kernel function header.
 */
#ifndef DGL_ARRAY_CUDA_SPMM_CUH_
#define DGL_ARRAY_CUDA_SPMM_CUH_

#include "bcast.h"
#include "cuda_common.h"
#include "macro.cuh"
#include "utils.h"

#include "../ops.cuh"

namespace dgl {

using namespace cuda;

namespace aten {
namespace cuda {

/*!
 * \brief CUDA kernel of g-SpMM on Csr format.
 * \note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
          bool UseBcast = false>
__global__ void
SpMMCsrKernel(const DType *__restrict__ ufeat, const DType *__restrict__ efeat,
              DType *__restrict__ out, Idx *__restrict__ arg_u,
              Idx *__restrict__ arg_e, const Idx *__restrict__ indptr,
              const Idx *__restrict__ indices, const Idx *__restrict__ edge_map,
              int64_t num_rows, int64_t num_cols,
              const int64_t *__restrict__ ubcast_off,
              const int64_t *__restrict__ ebcast_off, int64_t ufeat_len,
              int64_t efeat_len, int64_t out_len) {
  // SPMM with CSR.
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  const int stride_x = blockDim.x * gridDim.x;
  while (ty < num_rows) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      DType local_accum = ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = _ldg(edge_map + i);
        const Idx cid = _ldg(indices + i);
        const DType *uoff =
            BinaryOp::use_lhs ? (ufeat + cid * ufeat_len) : nullptr;
        const DType *eoff =
            BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
        DType out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(&local_accum, &local_argu, &local_arge, out, cid, eid);
      }
      out[ty * out_len + tx] = local_accum;
      if (ReduceOp::require_arg && BinaryOp::use_lhs)
        arg_u[ty * out_len + tx] = local_argu;
      if (ReduceOp::require_arg && BinaryOp::use_rhs)
        arg_e[ty * out_len + tx] = local_arge;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/*!
 * \brief CUDA implementation of g-SpMM on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer. \param arge Arg-Min/Max on edges. which refers the source node
 * indices correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCsr(BcastOff bcast, int64_t num_rows, int64_t num_cols,
             at::Tensor csr_indptr, at::Tensor csr_indices, at::Tensor csr_data,
             at::Tensor ufeat, at::Tensor efeat, at::Tensor out,
             at::Tensor argu, at::Tensor arge) {
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len,
          reduce_dim = bcast.reduce_size;
  at::Tensor ubcast_off = bcast.lhs_offset, ebcast_off = bcast.rhs_offset;
  bool use_bcast = bcast.use_bcast;

  const Idx *indptr = csr_indptr.data_ptr<Idx>();
  const Idx *indices = csr_indices.data_ptr<Idx>();
  const Idx *edge_map = csr_data.data_ptr<Idx>();
  const DType *ufeat_data = ufeat.data_ptr<DType>();
  const DType *efeat_data = efeat.data_ptr<DType>();
  DType *out_data = out.data_ptr<DType>();
  Idx *argu_data = argu.data_ptr<Idx>();
  Idx *arge_data = arge.data_ptr<Idx>();

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((num_rows + nty - 1) / nty);
  // LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", "
  // << nty << ")";
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  BCAST_SWITCH(use_bcast, UseBcast,
               {CUDA_KERNEL_CALL(
                   (SpMMCsrKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast>),
                   nblks, nthrs, 0, nullptr, ufeat_data, efeat_data, out_data,
                   argu_data, arge_data, indptr, indices, edge_map, num_rows,
                   num_cols, ubcast_off.data_ptr<int64_t>(),
                   ebcast_off.data_ptr<int64_t>(), lhs_len, rhs_len, len)});
}

} // namespace cuda
} // namespace aten
} // namespace dgl

#endif
