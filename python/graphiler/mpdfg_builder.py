import sys
import inspect
from pathlib import Path
from importlib import reload

import torch
from graphiler import EdgeBatchDummy, NodeBatchDummy, optimizer
from graphiler.mpdfg import builder, MPDFGAnnotation

# load DGLGraph and my_ops
DGL_PATH = str(Path.home()) + "/.dgl/"
sys.path.append(DGL_PATH)
torch.classes.load_library(DGL_PATH + "libgraphiler.so")
torch.ops.load_library(DGL_PATH + "libgraphiler.so")

FuncTemplate = r'''
import torch
from typing import Dict
from pathlib import Path
torch.classes.load_library(str(Path.home()) + "/.dgl/libgraphiler.so")

def mpdfg_func(dglgraph: torch.classes.my_classes.DGLGraph, 
                ndata: Dict[str, torch.Tensor], edata: Dict[str, torch.Tensor],
                ntypedata: Dict[str, torch.Tensor], etypedata: Dict[str, torch.Tensor], __extra__):
    return {'h':torch.tensor(0)}
'''


class MPDFG():
    def __init__(self, func) -> None:
        self.forward = func
        self.annotations = MPDFGAnnotation(func.graph)


def mpdfg_builder(msg_func, reduce_func, update_func=None, opt_level=2):
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

    mpdfg_func = FuncTemplate.replace(
        "__extra__", ",".join(str(i) for i in extra_params))
    with open(DGL_PATH + "mpdfg_temp.py", 'w') as f:
        f.write(mpdfg_func)

    # import source code template for scripting
    import mpdfg_temp
    reload(mpdfg_temp)
    mpdfg_func = torch.jit.script(mpdfg_temp.mpdfg_func)
    mpdfg = MPDFG(mpdfg_func)

    # Todo: inline or not inline?
    msg_func = torch.jit.script(msg_func).inlined_graph
    reduce_func = torch.jit.script(reduce_func).inlined_graph
    update_func = torch.jit.script(
        update_func).inlined_graph if update_func else None

    print("UDF message function:\n", msg_func)
    print("UDF reduce function:\n", reduce_func)
    print("UDF update function:\n", update_func)

    builder(mpdfg.annotations, msg_func, reduce_func, update_func)
    optimizer(mpdfg.annotations, opt_level)
    print("MPDFG:\n", mpdfg.forward.graph)

    return mpdfg
