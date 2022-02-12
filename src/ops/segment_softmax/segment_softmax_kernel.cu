#include "../ops.cuh"

// feature dim = 1
__global__ void softmax_v(const float *features, const int *pointer,
                          const int *indices, float *__restrict__ next_layer) {
  int neighbor_offset = pointer[blockIdx.x];
  int degree = pointer[blockIdx.x + 1] - neighbor_offset;

  float max_local = 0.0f;
  for (int i = 0; i < degree / 32; i++) {
    max_local = max(features[indices[neighbor_offset + i * 32 + threadIdx.x]],
                    max_local);
  }
  if (threadIdx.x < degree % 32) {
    max_local = max(features[indices[neighbor_offset + degree - (degree % 32) +
                                     threadIdx.x]],
                    max_local);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    max_local = max(__shfl_down_sync(FULL_MASK, max_local, offset), max_local);
  }
  max_local = __shfl_sync(FULL_MASK, max_local, 0);

  float exp_local = 0.0f;
  for (int i = 0; i < degree / 32; i++) {
    exp_local += expf(
        features[indices[neighbor_offset + i * 32 + threadIdx.x]] - max_local);
  }
  if (threadIdx.x < degree % 32) {
    exp_local += expf(features[indices[neighbor_offset + degree -
                                       (degree % 32) + threadIdx.x]] -
                      max_local);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    exp_local += __shfl_down_sync(FULL_MASK, exp_local, offset);
  }
  exp_local = __shfl_sync(FULL_MASK, exp_local, 0);

  for (int i = 0; i < degree / 32; i++) {
    int neighbor = indices[neighbor_offset + i * 32 + threadIdx.x];
    next_layer[neighbor] = expf(features[neighbor] - max_local) / exp_local;
  }
  if (threadIdx.x < degree % 32) {
    int neighbor =
        indices[neighbor_offset + degree - (degree % 32) + threadIdx.x];
    next_layer[neighbor] = expf(features[neighbor] - max_local) / exp_local;
  }
  return;
}

// feature dim > 1, needed for multi head attention
__global__ void softmax_m(const float *features, const int *pointer,
                          const int *indices, float *__restrict__ next_layer) {}