# Graphiler

Reorganizing Graphiler for code release and artifact evaluation, still in construction so stay tuned!

To build and run (Install PyTorch and DGL prior to this):
```bash
git clone https://github.com/xiezhq-hermann/graphiler.git
cd graphiler
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
make
mkdir -p ~/.dgl
mv libgraphiler.so ~/.dgl/
cd ..
python setup.py install

python examples/GAT.py cora 1433
```