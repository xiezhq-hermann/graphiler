import time
import numpy as np
import pandas as pd

from torch.cuda import profiler, synchronize, max_memory_allocated, reset_peak_memory_stats


def check_equal(first, second):
    np.testing.assert_allclose(first.cpu().detach(
    ).numpy(), second.cpu().detach().numpy(), rtol=1e-3)
    print("corectness check passed!")


def bench(net, net_params, tag="", nvprof=False, memory=False, steps=1001, prof_df=None):
    # warm up
    for i in range(5):
        net(*net_params)
    synchronize()
    reset_peak_memory_stats()
    if nvprof:
        profiler.start()
    start_time = time.time()
    for i in range(steps):
        logits = net(*net_params)
    synchronize()
    if nvprof:
        profiler.stop()
    elapsed_time = (time.time() - start_time) / steps * 1000
    print("{} elapsed time: {} ms/infer".format(tag, elapsed_time))
    if prof_df is not None:
        prof_df[tag, "elapsed_time"] = elapsed_time 
    if memory:
        max_mem_consumption = max_memory_allocated() / 1048576
        print("max memory consumption: {} MB".format(max_mem_consumption))
        prof_df[tag, "mem"] = max_mem_consumption
    return logits


def init_df(datasets, tags, metrics):
    index = pd.MultiIndex.from_product(
        [datasets, tags, metrics],
        names=["dataset", "tag", "metric"]
    )
    return pd.Series(np.zeros(len(datasets), len(tags), len(metrics)), index=index)
 