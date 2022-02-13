from graphiler.mpdfg import split


def optimizer(mpdfg, opt_level):
    if opt_level == 0:
        return
    if opt_level > 0:
        split(mpdfg)
    if opt_level > 1:
        pass
    return
