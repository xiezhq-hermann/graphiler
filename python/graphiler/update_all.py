def update_all(g, mpdfg, msg_params=(), reduce_params=(), update_params=()):
    # Todo: ntype and etype data format defination
    updated = mpdfg.forward(g.DGLGraph, g.ndata, g.edata, {}, {}, *msg_params,
                            *reduce_params, *update_params)
    for key in updated:
        g.ndata[key] = updated[key]
