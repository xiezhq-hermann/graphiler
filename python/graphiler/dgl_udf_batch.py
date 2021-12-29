import torch
from typing import Dict


@torch.jit.script
class EdgeBatchDummy(object):
    def __init__(self, src_data: Dict[str, torch.Tensor], edge_data: Dict[str, torch.Tensor], dst_data: Dict[str, torch.Tensor], srctype_data: Dict[str, torch.Tensor], edgetype_data: Dict[str, torch.Tensor], dsttype_data: Dict[str, torch.Tensor]):
        self._src_data = src_data
        self._edge_data = edge_data
        self._dst_data = dst_data
        self._srctype_data = srctype_data
        self._edgetype_data = edgetype_data
        self._dsttype_data = dsttype_data

    @property
    def src(self):
        return self._src_data

    @property
    def dst(self):
        return self._dst_data

    @property
    def data(self):
        return self._edge_data

    @property
    def srctype(self):
        return self._srctype_data

    @property
    def dsttype(self):
        return self._dsttype_data

    @property
    def type(self):
        return self._edgetype_data


@torch.jit.script
class NodeBatchDummy(object):
    def __init__(self, msgs: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor], nodetype_data: Dict[str, torch.Tensor]):
        self._msgs = msgs
        self._node_data = data
        self._nodetype_data = nodetype_data

    @property
    def mailbox(self):
        return self._msgs

    @property
    def data(self):
        return self._node_data

    @property
    def type(self):
        return self._nodetype_data
