FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV NV_CUDA_LIB_VERSION "11.1.1-1"

FROM base as base-amd64

ENV NV_CUDA_CUDART_DEV_VERSION 11.1.74-1
ENV NV_NVML_DEV_VERSION 11.1.74-1
ENV NV_LIBCUSPARSE_DEV_VERSION 11.3.0.10-1
ENV NV_LIBNPP_DEV_VERSION 11.1.2.301-1
ENV NV_LIBNPP_DEV_PACKAGE libnpp-dev-11-1=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_VERSION 11.3.0.106-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME libcublas-dev-11-1
ENV NV_LIBCUBLAS_DEV_PACKAGE ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION 2.8.4-1
ENV NCCL_VERSION 2.8.4-1
ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda11.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget\
    git-all\
    cmake\
    libtinfo5 libncursesw5 \
    cuda-cudart-dev-11-1=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-11-1=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-11-1=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-11-1=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-11-1=${NV_NVML_DEV_VERSION} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    libcusparse-dev-11-1=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# cudnn

ENV NV_CUDNN_VERSION 8.0.5.39

ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda11.1"
ENV NV_CUDNN_PACKAGE_DEV "libcudnn8-dev=$NV_CUDNN_VERSION-1+cuda11.1"
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} && \
    rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME}

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# install miniconda
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-py38_4.10.3-Linux-x86_64.sh

# install dependent packages
ENV PATH="/opt/conda/bin:$PATH"
RUN pip install torch==1.8.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html\
    && pip install dgl-cu111==0.6.1\
       torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.8.0+cu111.html\
    && ogb==1.3.2

# git clone and build graphiler
# avoid docker from caching the git repo
ADD https://api.github.com/repos/xiezhq-hermann/graphiler/git/refs/heads/main version.json
RUN git clone -b main https://github.com/xiezhq-hermann/graphiler.git /root/graphiler
ENV Torch_DIR /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch
WORKDIR /root/graphiler
RUN mkdir -p /root/.dgl
RUN mkdir build \
  && cd build \
  && cmake .. \
  && make -j \
  && mv libgraphiler.so /root/.dgl \
  && cd .. \
  && python setup.py install
ENV DGLBACKEND pytorch
