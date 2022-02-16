# Graphiler

Reorganizing Graphiler for code release and artifact evaluation, still in construction so stay tuned!

# Install and run

```bash
pip install -r requirements.txt
git clone https://github.com/xiezhq-hermann/graphiler.git
cd graphiler
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
make
mkdir -p ~/.dgl
mv libgraphiler.so ~/.dgl/
cd ..
python setup.py install

# create directory storing outputs
mkdir -p output

# benchmark all GAT implementation on all datasets
python examples/GAT/GAT.py all 0
./visualize.sh GAT

# run all experiments and visualize results
export REPEAT=50  # manually specify the number of repeats, you can change it to whatever you want.
./run_all.sh
```

# Run within docker

The simplest way to reproduce the artifact is to use docker.

## Setup

Install docker and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Build and run

You can choose to build the image from scratch.

```
docker build -f docker/Dockerfile -t graphiler .
```

Or direct pull an pre-built image from docker hub:

```
docker pull expye/graphiler-ae:v0.1
docker tag expye/graphiler-ae:v0.1 graphiler
```

## Run experiments

```
docker run --gpus all -i -t -v $(pwd)/output:/root/graphiler/output graphiler ./run_all.sh
```

Note: The number of repeats in docker was set to `50` by default.
