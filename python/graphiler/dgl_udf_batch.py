import torch
from typing import Dict


@torch.jit.script
class EdgeBatchDummy(object):
    def __init__(self, src_data: Dict[str, torch.Tensor], edge_data: Dict[str, torch.Tensor], dst_data: Dict[str, torch.Tensor]):
        self._src_data = src_data
        self._edge_data = edge_data
        self._dst_data = dst_data

    @property
    def src(self):
        return self._src_data

    @property
    def dst(self):
        return self._dst_data

    @property
    def data(self):
        return self._edge_data


@torch.jit.script
class NodeBatchDummy(object):
    def __init__(self, data: Dict[str, torch.Tensor], msgs: Dict[str, torch.Tensor]):
        self._data = data
        self._msgs = msgs

    @property
    def data(self):
        return self._data

    @property
    def mailbox(self):
        return self._msgs


# @torch.jit.script
# class DGLGraph():
#     def __init__(self, node_data: Dict[str, torch.Tensor], edge_data: Dict[str, torch.Tensor]):
#         self._node_data = node_data
#         self._edge_data = edge_data

#     @property
#     def ndata(self):
#         return self._node_data

#     @property
#     def edata(self):
#         return self._edge_data
