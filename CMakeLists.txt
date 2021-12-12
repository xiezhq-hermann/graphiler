cmake_minimum_required(VERSION 3.15.0)
project(graphiler)

find_package(Torch REQUIRED)

include_directories("include")

# Define our library target
add_library(graphiler SHARED
    src/mpdfg.cpp)

# Enable C++14
target_compile_features(graphiler PRIVATE cxx_std_14)

# Link against LibTorch
target_link_libraries(graphiler "${TORCH_LIBRARIES}")