#include "../ops.cuh"

template <typename T>
__global__ void scatter(const int num_nodes, const int feat_dim,
                        const T *features, const int *pointers,
                        const int *indices, T *__restrict__ next_layer) {
  int node_id = blockDim.y * blockIdx.x + threadIdx.y;
  int feat_id = threadIdx.x + blockDim.x * blockIdx.y;
  if (node_id >= num_nodes || feat_id >= feat_dim)
    return;

  T local = features[node_id * feat_dim + feat_id];

  int target;
  int start = pointers[node_id];
  int end = pointers[node_id + 1];
  for (int i = start; i < end; i++) {
    target = indices[i] * feat_dim + feat_id;
    next_layer[target] = local;
  }
}

template __global__ void scatter<float>(const int num_nodes, const int feat_dim,
                                        const float *features,
                                        const int *pointers, const int *indices,
                                        float *__restrict__ next_layer);
template __global__ void
scatter<int64_t>(const int num_nodes, const int feat_dim,
                 const int64_t *features, const int *pointers,
                 const int *indices, int64_t *__restrict__ next_layer);