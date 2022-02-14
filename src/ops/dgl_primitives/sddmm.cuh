/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/sddmm.cuh
 * \brief SDDMM CUDA kernel function header.
 */
#ifndef DGL_ARRAY_CUDA_SDDMM_CUH_
#define DGL_ARRAY_CUDA_SDDMM_CUH_

#include "bcast.h"
#include "cuda_common.h"
#include "functor.cuh"
#include "macro.cuh"
#include "selector.h"
#include "utils.h"

#include "../ops.cuh"

namespace dgl {

using namespace cuda;

namespace aten {
namespace cuda {

constexpr unsigned int full_mask = 0xffffffff;

/*!
 * \brief CUDA kernel of g-SDDMM on Coo format.
 * \note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension.
 */
template <typename Idx, typename DType, typename BinaryOp,
          bool UseBcast = false, int LhsTarget = 0, int RhsTarget = 2>
__global__ void
SDDMMCooKernel(const DType *__restrict__ lhs, const DType *__restrict__ rhs,
               DType *__restrict__ out, const Idx *__restrict__ row,
               const Idx *__restrict__ col, int64_t N, int64_t M, int64_t E,
               int64_t reduce_size, const int64_t *__restrict__ lhs_off,
               const int64_t *__restrict__ rhs_off, int64_t lhs_len,
               int64_t rhs_len, int64_t out_len) {
  // SDDMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = ty;
    const DType *lhsoff =
        BinaryOp::use_lhs
            ? (lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len)
            : nullptr;
    const DType *rhsoff =
        BinaryOp::use_rhs
            ? (rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len)
            : nullptr;
    DType *outoff = out + eid * out_len;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_x = blockDim.x * gridDim.x;
    while (tx < out_len) {
      const Idx lhs_add = UseBcast ? lhs_off[tx] : tx;
      const Idx rhs_add = UseBcast ? rhs_off[tx] : tx;
      DType val = BinaryOp::Call(lhsoff + lhs_add * reduce_size,
                                 rhsoff + rhs_add * reduce_size, reduce_size);
      outoff[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/*!
 * \brief CUDA kernel of SDDMM-dot on Coo format, accelerated with tree
 * reduction. \note it uses edge parallel strategy, different threadblocks (on
 * y-axis) is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension.
 */
template <typename Idx, typename DType, bool UseBcast = false,
          int LhsTarget = 0, int RhsTarget = 2>
__global__ void SDDMMCooTreeReduceKernel(
    const DType *__restrict__ lhs, const DType *__restrict__ rhs,
    DType *__restrict__ out, const Idx *__restrict__ row,
    const Idx *__restrict__ col, int64_t N, int64_t M, int64_t E,
    int64_t reduce_size, const int64_t *__restrict__ lhs_off,
    const int64_t *__restrict__ rhs_off, int64_t lhs_len, int64_t rhs_len,
    int64_t out_len) {
  Idx ty = blockIdx.x * blockDim.y + threadIdx.y;
  if (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = ty;
    const DType *lhsoff =
        lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len;
    const DType *rhsoff =
        rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len;
    DType *outoff = out + eid * out_len;
    int tx = threadIdx.x; // tx < 32
    for (int i = blockIdx.y; i < out_len;
         i += gridDim.y) { // over output feature dimension
      const Idx lhs_add = UseBcast ? __ldg(lhs_off + i) : i;
      const Idx rhs_add = UseBcast ? __ldg(rhs_off + i) : i;
      DType val = reduce::Sum<Idx, DType>::zero();
      ;
      for (int j = tx; j < reduce_size; j += 64) {
        val += lhsoff[lhs_add * reduce_size + j] *
               rhsoff[rhs_add * reduce_size + j];
        if (j + 32 < reduce_size)
          val += lhsoff[lhs_add * reduce_size + j + 32] *
                 rhsoff[rhs_add * reduce_size + j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(full_mask, val, offset);
      if (tx == 0)
        outoff[i] = val;
    }
  }
}

// Binary search the row_offsets to find the source node of the edge id.
template <typename Idx>
__device__ __forceinline__ Idx BinarySearchSrc(const Idx *array, Idx length,
                                               Idx eid) {
  Idx lo = 0, hi = length - 1;
  while (lo < hi) {
    Idx mid = (lo + hi) >> 1;
    if (_ldg(array + mid) <= eid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // INVARIANT: lo == hi
  if (_ldg(array + hi) == eid) {
    return hi;
  } else {
    return hi - 1;
  }
}

/*!
 * \brief CUDA kernel of g-SDDMM on Csr format.
 * \note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension. To efficiently find the source node idx and
 * destination node index of an given edge on Csr format, it uses binary search
 * (time complexity O(log N)).
 */
template <typename Idx, typename DType, typename BinaryOp,
          bool UseBcast = false, int LhsTarget = 0, int RhsTarget = 2>
__global__ void SDDMMCsrKernel(
    const DType *__restrict__ lhs, const DType *__restrict__ rhs,
    DType *__restrict__ out, const Idx *__restrict__ indptr,
    const Idx *__restrict__ indices, const Idx *__restrict__ edge_map,
    int64_t N, int64_t M, int64_t E, int64_t reduce_size,
    const int64_t *__restrict__ lhs_off, const int64_t *__restrict__ rhs_off,
    int64_t lhs_len, int64_t rhs_len, int64_t out_len) {
  // SDDMM with Csr.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = BinarySearchSrc<Idx>(indptr, N + 1, ty);
    const Idx dst = _ldg(indices + ty);
    const Idx eid = _ldg(edge_map + ty);
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType *lhsoff =
        BinaryOp::use_lhs
            ? (lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len)
            : nullptr;
    const DType *rhsoff =
        BinaryOp::use_rhs
            ? (rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len)
            : nullptr;
    DType *outoff = out + eid * out_len;
    while (tx < out_len) {
      const Idx lhs_add = UseBcast ? lhs_off[tx] : tx;
      const Idx rhs_add = UseBcast ? rhs_off[tx] : tx;
      DType val = BinaryOp::Call(lhsoff + lhs_add * reduce_size,
                                 rhsoff + rhs_add * reduce_size, reduce_size);
      outoff[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename Idx, typename DType, bool UseBcast = false,
          int LhsTarget = 0, int RhsTarget = 2>
__global__ void SDDMMCsrTreeReduceKernel(
    const DType *__restrict__ lhs, const DType *__restrict__ rhs,
    DType *__restrict__ out, const Idx *__restrict__ indptr,
    const Idx *__restrict__ indices, const Idx *__restrict__ edge_map,
    int64_t N, int64_t M, int64_t E, int64_t reduce_size,
    const int64_t *__restrict__ lhs_off, const int64_t *__restrict__ rhs_off,
    int64_t lhs_len, int64_t rhs_len, int64_t out_len) {
  Idx ty = blockIdx.x * blockDim.y + threadIdx.y;
  if (ty < E) {
    const Idx src = BinarySearchSrc<Idx>(indptr, N + 1, ty);
    const Idx dst = _ldg(indices + ty);
    const Idx eid = _ldg(edge_map + ty);
    const DType *lhsoff =
        lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len;
    const DType *rhsoff =
        rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len;
    DType *outoff = out + eid * out_len;
    int tx = threadIdx.x; // tx < 32
    for (int i = blockIdx.y; i < out_len;
         i += gridDim.y) { // over output feature dimension
      const Idx lhs_add = UseBcast ? __ldg(lhs_off + i) : i;
      const Idx rhs_add = UseBcast ? __ldg(rhs_off + i) : i;
      DType val = reduce::Sum<Idx, DType>::zero();
      ;
      for (int j = tx; j < reduce_size; j += 64) {
        val += lhsoff[lhs_add * reduce_size + j] *
               rhsoff[rhs_add * reduce_size + j];
        if (j + 32 < reduce_size)
          val += lhsoff[lhs_add * reduce_size + j + 32] *
                 rhsoff[rhs_add * reduce_size + j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(full_mask, val, offset);
      if (tx == 0)
        outoff[i] = val;
    }
  }
}

/*!
 * \brief CUDA implementation of g-SDDMM on Coo format.
 */
template <typename Idx, typename DType, typename Op, int LhsTarget = 0,
          int RhsTarget = 2>
void SDDMMCoo(BcastOff bcast, int64_t num_rows, int64_t num_cols,
              at::Tensor coo_row, at::Tensor coo_col, at::Tensor lhs,
              at::Tensor rhs, at::Tensor out) {
  const Idx *row = coo_row.data_ptr<Idx>();
  const Idx *col = coo_col.data_ptr<Idx>();
  const DType *lhs_data = lhs.data_ptr<DType>();
  const DType *rhs_data = rhs.data_ptr<DType>();
  DType *out_data = out.data_ptr<DType>();
  int64_t nnz = coo_row.numel();
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len,
          reduce_dim = bcast.reduce_size;
  at::Tensor lhs_off = bcast.lhs_offset, rhs_off = bcast.rhs_offset;

  bool use_bcast = bcast.use_bcast;
  if (std::is_same<Op, binary::Dot<DType>>::value && reduce_dim >= 32) {
    const int ntx = 32; // on feature dimension
    const int nty = 8;  // on out dimension
    const int nbx = (nnz + nty - 1) / nty;
    const int nby = FindNumBlocks<'y'>(len);
    const dim3 nblks(nbx, nby);
    const dim3 nthrs(ntx, nty);
    BCAST_SWITCH(use_bcast, UseBcast, {
      CUDA_KERNEL_CALL((SDDMMCooTreeReduceKernel<Idx, DType, UseBcast,
                                                 LhsTarget, RhsTarget>),
                       nblks, nthrs, 0, nullptr, lhs_data, rhs_data, out_data,
                       row, col, num_rows, num_cols, nnz, reduce_dim,
                       lhs_off.data_ptr<int64_t>(), rhs_off.data_ptr<int64_t>(),
                       lhs_len, rhs_len, len);
    });
  } else {
    const int ntx = FindNumThreads(len);
    const int nty = CUDA_MAX_NUM_THREADS / ntx;
    const int nbx = (len + ntx - 1) / ntx;
    const int nby = FindNumBlocks<'y'>((nnz + nty - 1) / nty);
    const dim3 nblks(nbx, nby);
    const dim3 nthrs(ntx, nty);
    BCAST_SWITCH(use_bcast, UseBcast, {
      CUDA_KERNEL_CALL(
          (SDDMMCooKernel<Idx, DType, Op, UseBcast, LhsTarget, RhsTarget>),
          nblks, nthrs, 0, nullptr, lhs_data, rhs_data, out_data, row, col,
          num_rows, num_cols, nnz, reduce_dim, lhs_off.data_ptr<int64_t>(),
          rhs_off.data_ptr<int64_t>(), lhs_len, rhs_len, len);
    });
  }
}

/*!
 * \brief CUDA implementation of g-SDDMM on Csr format.
 */
template <typename Idx, typename DType, typename Op, int LhsTarget = 0,
          int RhsTarget = 2>
void SDDMMCsr(BcastOff bcast, int64_t num_rows, int64_t num_cols,
              at::Tensor csr_indptr, at::Tensor csr_indices,
              at::Tensor csr_data, at::Tensor lhs, at::Tensor rhs,
              at::Tensor out) {
  const Idx *indptr = csr_indptr.data_ptr<Idx>();
  const Idx *indices = csr_indices.data_ptr<Idx>();
  const Idx *edge_map = csr_data.data_ptr<Idx>();
  const DType *lhs_data = lhs.data_ptr<DType>();
  const DType *rhs_data = rhs.data_ptr<DType>();
  DType *out_data = out.data_ptr<DType>();
  int64_t nnz = csr_indices.numel();
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len,
          reduce_dim = bcast.reduce_size;
  at::Tensor lhs_off = bcast.lhs_offset, rhs_off = bcast.rhs_offset;

  bool use_bcast = bcast.use_bcast;
  if (std::is_same<Op, binary::Dot<DType>>::value && reduce_dim >= 32) {
    const int ntx = 32; // on feature dimension
    const int nty = 8;  // on out dimension
    const int nbx = (nnz + nty - 1) / nty;
    const int nby = FindNumBlocks<'y'>(len);
    const dim3 nblks(nbx, nby);
    const dim3 nthrs(ntx, nty);
    BCAST_SWITCH(use_bcast, UseBcast, {
      CUDA_KERNEL_CALL((SDDMMCsrTreeReduceKernel<Idx, DType, UseBcast,
                                                 LhsTarget, RhsTarget>),
                       nblks, nthrs, 0, nullptr, lhs_data, rhs_data, out_data,
                       indptr, indices, edge_map, num_rows, num_cols, nnz,
                       reduce_dim, lhs_off.data_ptr<int64_t>(),
                       rhs_off.data_ptr<int64_t>(), lhs_len, rhs_len, len);
    });
  } else {
    const int ntx = FindNumThreads(len);
    const int nty = CUDA_MAX_NUM_THREADS / ntx;
    const int nbx = (len + ntx - 1) / ntx;
    const int nby = FindNumBlocks<'y'>((nnz + nty - 1) / nty);
    const dim3 nblks(nbx, nby);
    const dim3 nthrs(ntx, nty);
    BCAST_SWITCH(use_bcast, UseBcast, {
      CUDA_KERNEL_CALL(
          (SDDMMCsrKernel<Idx, DType, Op, UseBcast, LhsTarget, RhsTarget>),
          nblks, nthrs, 0, nullptr, lhs_data, rhs_data, out_data, indptr,
          indices, edge_map, num_rows, num_cols, nnz, reduce_dim,
          lhs_off.data_ptr<int64_t>(), rhs_off.data_ptr<int64_t>(), lhs_len,
          rhs_len, len);
    });
  }
}

} // namespace cuda
} // namespace aten
} // namespace dgl

#endif