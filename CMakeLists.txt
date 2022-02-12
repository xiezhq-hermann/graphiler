cmake_minimum_required(VERSION 3.15.0)
project(graphiler LANGUAGES CUDA CXX)

find_package(Torch REQUIRED)
# find_package(PythonInterp REQUIRED)

include_directories("include")
include_directories("src/ops")
include_directories("src/ops/dgl_primitives")

# Define our library target
add_library(graphiler SHARED
    src/dglgraph.cpp
    src/ops/broadcast/broadcast.cu
    src/ops/broadcast/scatter_kernel.cu
    src/ops/segment_softmax/segment_softmax.cu
    src/ops/segment_softmax/segment_softmax_kernel.cu
    # note dgl_primitives requires CUDA version >= 11
    src/ops/dgl_primitives/spmm.cu
    src/ops/dgl_primitives/bcast.cc
    )
    
# Todo: more architectures
set_target_properties(graphiler PROPERTIES CUDA_ARCHITECTURES "70")

# Enable C++14
target_compile_features(graphiler PRIVATE cxx_std_14)

# Link against LibTorch
target_link_libraries(graphiler ${TORCH_LIBRARIES} ${CUDA_cusparse_LIBRARY})