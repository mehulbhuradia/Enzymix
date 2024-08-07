from torch import nn
import torch
from diffab.modules.common.layers import PositionalEncoding

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,x_dim=9, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False, additional_layers=0):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.x_dim=x_dim
        edge_coords_nf = 1
        
        edge_mlp_layers = []
        edge_mlp_layers.append(nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf))
        edge_mlp_layers.append(act_fn)
        for i in range(additional_layers):
            edge_mlp_layers.append(nn.Linear(hidden_nf, hidden_nf))
            edge_mlp_layers.append(act_fn)
        edge_mlp_layers.append(nn.Linear(hidden_nf, hidden_nf))
        edge_mlp_layers.append(act_fn)

        self.edge_mlp = nn.Sequential(*edge_mlp_layers)

        node_mlp_layers = []
        node_mlp_layers.append(nn.Linear(hidden_nf + input_nf, hidden_nf))
        node_mlp_layers.append(act_fn)
        for i in range(additional_layers):
            node_mlp_layers.append(nn.Linear(hidden_nf, hidden_nf))
            node_mlp_layers.append(act_fn)
        node_mlp_layers.append(nn.Linear(hidden_nf, output_nf))
        self.node_mlp = nn.Sequential(*node_mlp_layers)

        coord_mlp_layers = []
        coord_mlp_layers.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp_layers.append(act_fn)

        for i in range(additional_layers):
            coord_mlp_layers.append(nn.Linear(hidden_nf, hidden_nf))
            coord_mlp_layers.append(act_fn)
        
        layer = nn.Linear(hidden_nf, x_dim*x_dim) #used to have bias=False
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001) # important for some reason, code breaks when i remove it
        coord_mlp_layers.append(layer)
        
        if self.tanh:
            coord_mlp_layers.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp_layers)

        if self.attention:
            att_mlp_layers = []
            for i in range(additional_layers):
                att_mlp_layers.append(nn.Linear(hidden_nf, hidden_nf))
                att_mlp_layers.append(act_fn)
            att_mlp_layers.append(nn.Linear(hidden_nf, 1))
            att_mlp_layers.append(nn.Sigmoid())
            self.att_mlp = nn.Sequential(*att_mlp_layers)

        
    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        phi_x=self.coord_mlp(edge_feat)
        phi_x = phi_x.view(phi_x.shape[0],self.x_dim,self.x_dim)
        coord_diff = coord_diff.unsqueeze(1)
        trans = torch.bmm(coord_diff,phi_x).squeeze(1)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord += agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf,x_dim=9, in_edge_nf=0, device='cuda:0', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False,additional_layers=0,coords_agg='mean'):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.pos_enc = PositionalEncoding(num_funcs=1)
        self.embedding_in = []
        self.embedding_out = []
        for i in range(0, n_layers):
            self.add_module("embedding_in_%d" % i, nn.Linear(in_node_nf, self.hidden_nf))
            # if i < 2:
            #     additional_layers=24
            # else:
            #     additional_layers=0
            self.add_module("egcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,x_dim=x_dim,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh,additional_layers=additional_layers,coords_agg=coords_agg))
            self.add_module("embedding_out_%d" % i, nn.Linear(self.hidden_nf, out_node_nf))
        self.to(self.device)

    def forward(self, h, x, t, edges, batch_size):
        for i in range(0, self.n_layers):
            res_len = h.shape[0]/batch_size.item()
            indeces=torch.arange(res_len).unsqueeze(-1)
            indeces = indeces.to(h.device)
            pos_e = self.pos_enc(indeces)
            pos_e = pos_e.to(h.device)
            pos_chunks = []
            for _ in range(batch_size):
                pos_chunks.append(pos_e)
            pos_h = torch.cat(pos_chunks, dim=0)
            h = torch.cat((h, pos_h), dim=1)
            h = torch.cat((h, t), dim=1)
            h = self._modules["embedding_in_%d" % i](h)
            
            h, x, _ = self._modules["egcl_%d" % i](h, edges, x)
            
            if (torch.isnan(x).any().item()):
                # print("NAN in x at layer %d" % i)
                raise KeyboardInterrupt()
            
            h = self._modules["embedding_out_%d" % i](h)

        return h, x



def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


if __name__ == "__main__":
    # Dummy parameters
    batch_size = 1
    n_nodes = 4
    n_feat = 1
    x_dim = 3

    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch_size *  n_nodes, n_feat)
    x = torch.ones(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    # Initialize EGNN
    egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1,x_dim=x_dim)

    # Run EGNN
    h, x = egnn(h, x, edges, edge_attr)
