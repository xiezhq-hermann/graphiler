FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget\
    git-all\
    cmake\
    gnuplot\
    ghostscript\
    texlive-font-utils\
    && rm -rf /var/lib/apt/lists/*

# install miniconda
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-py38_4.10.3-Linux-x86_64.sh

# install dependent packages
ENV PATH="/opt/conda/bin:$PATH"
RUN pip install torch==1.8.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html\
    && pip install dgl-cu111==0.6.1\
       torch-scatter==2.0.8 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.8.0+cu111.html\
    && pip install torch-geometric==2.0.1\
    && pip install ogb==1.3.2\
    && pip install pandas==1.3.5

# git clone and build graphiler
# avoid docker from caching the git repo
ADD https://api.github.com/repos/xiezhq-hermann/graphiler/git/refs/heads/main version.json
RUN git clone -b main https://github.com/xiezhq-hermann/graphiler.git /root/graphiler
WORKDIR /root/graphiler
RUN mkdir -p /root/.dgl
RUN mkdir build \
  && cd build \
  && cmake -DPYTHON_EXECUTABLE:FILEPATH="/opt/conda/bin/python" -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" .. \
  && make -j \
  && mv libgraphiler.so /root/.dgl \
  && cd .. \
  && python setup.py install
ENV DGLBACKEND pytorch
ENV REPEAT 50
ENV GRAPHILER /root/graphiler

# prefetch dataset
RUN python artifact/prefetch_dataset.py
# create output directory
RUN mkdir -p output
