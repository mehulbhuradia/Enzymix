from torch import nn
import torch
from torch_geometric.nn.conv import GATv2Conv


class GACNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_layers=4):
        self.in_node_nf=in_node_nf
        self.hidden_nf=hidden_nf
        self.out_node_nf=out_node_nf
        self.n_layers=n_layers
        super(GACNN, self).__init__()
        self.add_module("in_linear_layer", nn.Linear(self.in_node_nf, self.hidden_nf))
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GATv2Conv(self.hidden_nf,self.hidden_nf-3))
        self.add_module("out_linear_layer", nn.Linear(self.hidden_nf, self.out_node_nf))

    def forward(self, h, t, edges):
        h = torch.cat((h, t), dim=1)
        h = self._modules["in_linear_layer"](h)
        for i in range(0, self.n_layers):
            h = self._modules["gcl_%d" % i](h, edges)
            h = torch.cat((h, t), dim=1)
        h = self._modules["out_linear_layer"](h)
        return h