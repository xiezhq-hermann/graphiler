#pragma once

#include <torch/script.h>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "../../include/dglgraph.h"

#define FULL_MASK 0xffffffff

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    cudaError_t result = (x);                                                  \
    if (result != cudaSuccess) {                                               \
      const char *msg = cudaGetErrorString(result);                            \
      std::stringstream safe_call_ss;                                          \
      safe_call_ss << "\nerror: " #x " failed with error"                      \
                   << "\nfile: " << __FILE__ << "\nline: " << __LINE__         \
                   << "\nmsg: " << msg;                                        \
      throw std::runtime_error(safe_call_ss.str());                            \
    }                                                                          \
  } while (0)

template <typename T>
__global__ void scatter(const int num_nodes, const int feat_dim,
                        const T *features, const int *pointers,
                        const int *indices, T *__restrict__ next_layer);

__global__ void softmax_v(const float *features, const int *pointers,
                          const int *indices, float *__restrict__ next_layer);
__global__ void softmax_m(const float *features, const int *pointers,
                          const int *indices, float *__restrict__ next_layer);
