import sys
import inspect
from typing import Dict
from pathlib import Path

import torch
from graphiler import EdgeBatchDummy, NodeBatchDummy
from graphiler.mpdfg import builder

DGL_PATH = str(Path.home()) + "/.dgl/"
sys.path.append(DGL_PATH)
# torch.classes.load_library(
#     DGL_PATH + "libgraphiler.so")


FuncTemplate = r'''
import torch
from typing import Dict

def mpdfg_func(ndata: Dict[str, torch.Tensor], edata: Dict[str, torch.Tensor], __extra__):
    res = {'h': torch.ones(1)}
    return res
'''


class MPDFG():
    def __init__(self, func) -> None:
        self.forward = func
        self.annotations = torch.classes.my_classes.MPDFGAnnotation()


def mpdfg_builder(msg_func, reduce_func, update_func=None):
    extra_params = []

    def get_params(func, stage):
        for param in inspect.signature(func).parameters.values():
            if param.annotation in [EdgeBatchDummy, NodeBatchDummy]:
                continue
            extra_params.append(param.replace(name=param.name+stage))
        return
    get_params(msg_func, '_msg')
    get_params(reduce_func, '_reduce')
    if update_func:
        get_params(update_func, '_update')
        update_func = torch.jit.script(update_func).graph
    msg_func = torch.jit.script(msg_func).graph
    reduce_func = torch.jit.script(reduce_func).graph
    mpdfg_func = FuncTemplate.replace(
        "__extra__", ",".join(str(i) for i in extra_params))
    with open(DGL_PATH + "mpdfg_temp.py", 'w') as f:
        f.write(mpdfg_func)
    from mpdfg_temp import mpdfg_func
    mpdfg_func = torch.jit.script(mpdfg_func)
    mpdfg = MPDFG(mpdfg_func)
    builder(mpdfg.forward.graph, mpdfg.annotations,
            msg_func, reduce_func, update_func)
    return mpdfg
