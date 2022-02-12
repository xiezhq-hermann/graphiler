import time
import numpy as np

from torch.cuda import profiler, synchronize, max_memory_allocated, reset_peak_memory_stats


def check_equal(first, second):
    np.testing.assert_allclose(first.cpu().detach(
    ).numpy(), second.cpu().detach().numpy(), rtol=1e-3)
    print("corectness check passed!")


def bench(net, net_params, tag="", nvprof=False, memory=False, steps=1001):
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
    print("{} elapsed time: {} ms/infer".format(tag,
          (time.time() - start_time) / steps * 1000))
    if memory:
        print("max memory consumption: {} MB".format(
            max_memory_allocated()/1048576))
    return logits
