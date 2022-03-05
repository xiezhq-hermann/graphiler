# Graphiler
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6331245.svg)](https://doi.org/10.5281/zenodo.6331245)

Graphiler is a compiler stack built on top of DGL and TorchScript which compiles GNNs defined using user-defined functions (UDFs) into efficient execution plans. 
This allows creating high performance models while retaining the simplicity and expressiveness of the UDF interface.

## Repo Structure
``` bash
Graphiler
├── artifact                # scripts for running the artifact
├── docker
├── examples                # GNN models with different implementations
│   ├── GAT
│   ├── GCN
│   ├── HGT
│   └── RGCN
├── include
│   ├── dglgraph.h          # simplified Graph representation
│   └── mpdfg.h             # message passing data flow graph
├── python
│   ├── graphiler           # python wrapper
│   └── ...
└── src                     # source codes
    ├── builder             # MP-DFG builder
    ├── dglgraph.cpp
    ├── ops                 # graph primitives
    │   ├── broadcast
    │   ├── dgl_primitives
    │   │   ├── sddmm.cu
    │   │   ├── spmm.cu
    │   │   └── ...
    │   ├── segment_mm
    │   └── segment_softmax
    │   └── ...
    ├── optimizer           # optimization passes
    │   ├── dedup.cpp
    │   ├── fusion.cpp
    │   ├── optimizer.h
    │   ├── reorder.cpp
    │   └── split.cpp
    │   └── ...
    └── ...
```
## Build Graphiler and get started
### Play with docker
Docker is the easiest way to build the environment and reproduce the results. To make use of it, please make sure [docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) are properly installed and configured.

You can either build the image by yourself:
```
docker build -f docker/Dockerfile -t graphiler .
```
Or directly pull an pre-built image from docker hub:
```
docker pull expye/graphiler-ae:latest
docker tag expye/graphiler-ae:latest graphiler
```
To quickly verify the installation:
```
docker run --gpus all -i -t graphiler python examples/GAT/GAT.py pubmed 500
```
### Build from scratch
You can follow these instructions to build Graphiler on your machine:
```bash
# To install CUDA 11.1:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html 
git clone https://github.com/xiezhq-hermann/graphiler.git
cd graphiler
pip install -r requirements.txt # install PyTorch, DGL, PyG, etc
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
make
mkdir -p ~/.dgl
mv libgraphiler.so ~/.dgl/
cd ..
python setup.py install
# path used in scripts
export GRAPHILER=$(pwd)

# quick sanity check
python $GRAPHILER/examples/GAT/GAT.py pubmed 500
```

### Artifact Evaluation
Please go `artifact` directory for more information.
