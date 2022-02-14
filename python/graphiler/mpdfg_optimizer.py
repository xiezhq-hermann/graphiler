from graphiler.mpdfg import split, reorder


def optimizer(mpdfg, opt_level):
    if opt_level == 0:
        return
    if opt_level > 0:
        split(mpdfg)
        reorder(mpdfg)
    if opt_level > 1:
        pass
    return
