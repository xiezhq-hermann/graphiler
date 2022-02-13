import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, softmax


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed as
    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.
    Args:
        in_feats (int): Size of each input sample.
        out_feats (int): Size of each output sample.
        num_heads (int): Number of multi-head-attentions.
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        activation (callable, optional): Activation function to apply. (default: None)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 concat=True,
                 dropout=0.,
                 activation=None,
                 **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        self.lin = nn.Linear(in_feats, num_heads * out_feats, bias=False)

        self.att_i = nn.Parameter(torch.Tensor(1, num_heads, out_feats))
        self.att_j = nn.Parameter(torch.Tensor(1, num_heads, out_feats))

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)

    def forward(self, x, edge_index):
        if torch.is_tensor(x):
            x = self.dropout(x)
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.dropout(x[0]), self.dropout(x[1]))
            x = (self.lin(x[0]), self.lin(x[1]))

        out = self.propagate(edge_index, x=x)

        if self.activation is not None:
            out = self.activation(out)

        if not self.concat:
            out = out.view(-1, self._num_heads, self._out_feats).mean(dim=1)

        return out

    def message(self, x_i, x_j, edge_index_i, size_i):
        # Compute attention coefficients.
        x_i = x_i.view(-1, self._num_heads, self._out_feats)
        x_j = x_j.view(-1, self._num_heads, self._out_feats)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        # Sample attention coefficients stochastically.
        alpha = self.dropout(alpha)

        rst = x_j * alpha.view(-1, self._num_heads, 1)
        return rst.view(-1, self._num_heads * self._out_feats)


class GAT_PyG(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super(GAT_PyG, self).__init__()

        self.layer1 = GATConv(in_feats=in_dim,
                              out_feats=hidden_dim,
                              num_heads=1)

        self.layer2 = GATConv(in_feats=hidden_dim,
                              out_feats=out_dim,
                              num_heads=1)

    def reset_parameters(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

    def forward(self, x, adj):
        h = self.layer1(x, adj)
        h = F.elu(h)
        h = self.layer2(h, adj)
        return h
