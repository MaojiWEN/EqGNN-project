import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, None, :]
        return self.dropout(x)


class ESTAG(nn.Module):
    def __init__(self,  num_past, num_future, in_node_nf, in_edge_nf, hidden_nf, fft, eat, device, n_layers, n_nodes, nodes_att_dim=0,
                 act_fn=nn.SiLU(), coords_weight=1.0, with_mask=False, tempo=True, filter=True):
        super(ESTAG, self).__init__()
        self.hidden_nf = hidden_nf
        self.fft = fft
        self.eat = eat
        self.device = device
        self.n_layers = n_layers
        self.n_node  =n_nodes
        self.num_past = num_past
        self.tempo = tempo
        self.filter = filter

        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)

        for i in range(n_layers):
            self.add_module("egcl_%d" % ( i * 2 +1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=num_past-1, nodes_att_dim=nodes_att_dim,
                                                       act_fn=act_fn, recurrent=True, coords_weight=coords_weight, norm_diff=True, clamp=True))
            if self.eat:
                self.add_module("egcl_at_%d" % ( i * 2 +2), E_GCL_AT(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                                 act_fn=act_fn, recurrent=True, coords_weight=coords_weight, with_mask=with_mask))
        self.theta = nn.Parameter(torch.FloatTensor(num_future, num_past))

        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_nf, 1),
            nn.Sigmoid())

        self.reset_parameters()
        self.to(self.device)


    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)
        # O init
        self.theta.data*=0



    def FFT(self, h, x, n_nodes, edges):
        # x_ = rearrange(x, 't (b n) d -> t b n d', n=n_nodes)
        # x_bar = torch.mean(x_, dim=-2, keepdim=True)
        # x_norm = x_ - x_bar
        # x_norm = rearrange(x_norm, 't b n d -> (b n) d t')
        x_norm = rearrange(x, 't b d -> b d t')
        ### (b*n_node, 3, num_past)
        F = torch.fft.fftn(x_norm, dim=-1)

        ### (b*n_node, num_past-1)
        if self.filter:
            attn_val = self.attn_mlp(h[1:]).squeeze(-1).transpose(0, 1)
        else:
            # (b*n_node,), broadcast
            attn_val = torch.ones(h.shape[1], device=h.device).unsqueeze(-1)

        F = F[..., 1:]
        F_i = F[edges[0]]
        F_j = F[edges[1]]


        ## (n_egde, num_past-1)
        edge_attr = torch.abs(torch.sum(torch.conj(F_i) * F_j, dim=-2))

        edge_attr = edge_attr * (attn_val[edges[0]] * attn_val[edges[1]])

        edge_attr_norm = edge_attr / (torch.sum(edge_attr, dim=-1, keepdim=True ) +1e-9)

        ### (b*n_node, num_past-1)
        Fs = torch.abs(torch.sum( F**2, dim=-2))

        Fs = Fs * attn_val

        Fs_norm = Fs / (torch.sum(Fs, dim=-1, keepdim=True ) +1e-9)
        return edge_attr_norm, Fs_norm


    def forward(self, h, x, edges, edge_attr):
        """parameters
        former:
            h: (batch_num_nodes, num_past, feature_dim) - 节点特征，包含时序的速度标量特征
            x: (batch_num_nodes, num_past, coord_dim) - 节点位置特征
        Now:
            h: (num_past, b*n_node, 2)
            x: (num_past, b*n_node, 2)
            edges: (2, n_edge)
            edge_attr: (n_edge, 1)
        """
        h = h.permute(1, 0, 2)  # (num_past, batch_num_nodes, feature_dim)
        x = x.permute(1, 0, 2)  # (num_past, batch_num_nodes, coord_dim)

        # 2. 截取前 num_past 个时间片
        h = h[:self.num_past]  # -> [num_past, batch_num_nodes, 2]
        ### (num_past, b*n_node, hidden_nf)
        # h = self.embedding(h.unsqueeze(0).repeat(x.shape[0] ,1 ,1))
        h = self.embedding(h)

        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding
        # h = self.PosEmbedding(h)

        Fs =None
        if self.fft:
            ### (n_edge, num_past-1), (b*n_node, num_past-1)
            edge_attr, Fs = self.FFT(h, x, self.n_node, edges=edges)

        # print(edge_attr.shape)
        # print(Fs.shape)


        permute = lambda x: x.permute(1, 0, 2)
        h, x = map(permute, [h, x])
        # (b * n_node, num_past, dim)
        # print("h:", h.shape)
        # print("x:", x.shape)

        # FS [batch_num_nodes, num_past, num_past-1] E_A (n_edge, num_past, num_past-1)
        if Fs is not None: Fs = Fs.unsqueeze(1).repeat(1 ,h.shape[1] ,1)
        edge_attr = edge_attr.unsqueeze(1).repeat(1 ,h.shape[1] ,1)

        for i in range(self.n_layers):
            h, x = self._modules["egcl_%d" % ( i * 2 +1)](h, x, edges, edge_attr, Fs)

            if self.eat:
                h, x = self._modules["egcl_at_%d" % ( i * 2 +2)](h, x)

        x = permute(x)
        if self.tempo:
            x_hat =torch.einsum("ij,jkt->ikt", self.theta , x -x[-1].unsqueeze(0)).squeeze(0) +x[-1]
        else:
            x_hat =torch.einsum("ij,jkt->ikt", torch.softmax(self.theta ,dim=1) ,x).squeeze(0)

        return x_hat


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask) == 0


class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class GCL_basic(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(GCL_basic, self).__init__()

    def edge_model(self, source, target, edge_attr):
        pass

    def node_model(self, h, edge_index, edge_attr):
        pass

    def mpnn(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_feat)
        return x, edge_feat

    def forward(self, x, edge_index, edge_attr=None):
        xs = torch.stack([self.mpnn(x[i], edge_index, edge_attr)[0] for i in range(len(x))])
        return xs, None


class GCL(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=0, act_fn=nn.ReLU(), bias=True, attention=False,
                 t_eq=False, recurrent=True):
        super(GCL, self).__init__()
        self.attention = attention
        self.t_eq = t_eq
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid())

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias))

        # if recurrent:
        # self.gru = nn.GRUCell(hidden_nf, hidden_nf)

    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        out = self.edge_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out

    def node_model(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = out + h
            # out = self.gru(out, h)
        return out


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(),
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1

        self.adj_mlp = nn.Sequential(
            nn.Linear(edges_in_d, 1),
            act_fn,
        )

        # print(input_edge)
        # print(edge_coords_nf)
        # print(edges_in_d)
        # exit()
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)

        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, h, edge_index, radial, edge_attr, Fs):
        row, col = edge_index
        source, target = h[row], h[col]
        # source 的维度是 (n_edges, past_times, dim)。
        # target 的维度是 (n_edges, past_times, dim)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=-1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=-1)
        # print(source.shape)
        # print(target.shape)
        # print(radial.shape)
        # print(edge_attr.shape)
        # print("out_shape:", out.shape)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        # print("x: ", x.shape)
        # print("agg: ", agg.shape)
        # print("node_attr: ", node_attr.shape)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=-1)
        else:
            agg = torch.cat([x, agg], dim=-1)
        # print("agg: ", agg.shape)
        # exit()
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, Fs):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)  # * self.adj_mlp(edge_attr)#**

        trans = torch.clamp(trans, min=-100,
                            max=100)  # This is never activated but just in case it case it explosed it may save the train

        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))

        f = agg * self.coords_weight
        coord_ = coord + f
        return coord_, f

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, dim=-1, keepdim=True)

        if self.norm_diff:
            coord_diff = F.normalize(coord_diff, p=2, dim=-1)

        return radial, coord_diff

    def forward(self, h, coord, edge_index, edge_attr, Fs=None):
        radial, coord_diff = self.coord2radial(edge_index, coord)
        # print("radial coord:", radial.shape, coord_diff.shape)
        # radial :torch.Size([1077, 8, 1]) coord_diff: torch.Size([1077, 8, 2])
        # E_A (n_edge, num_past, num_past-1)
        edge_feat = self.edge_model(h, edge_index, radial, edge_attr, Fs=Fs)

        coord, _ = self.coord_model(coord, edge_index, coord_diff, edge_feat, Fs=Fs)  #

        h, _ = self.node_model(h, edge_index, edge_feat, node_attr=Fs)

        return h, coord


class E_GCL_AT(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False,
                 with_mask=False):
        super(E_GCL_AT, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.with_mask = with_mask
        self.hidden_nf = hidden_nf

        # edge_coords_nf = 1
        edge_coords_nf = 0

        self.q_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.k_mlp = nn.Sequential(
            nn.Linear(edge_coords_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.v_mlp = nn.Sequential(
            nn.Linear(edge_coords_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def forward(self, h, coord):
        """
            h: (bs*n_node, num_past, d)
            coord: (bs*n_node, num_past, 3)
        """
        ### (b*n_node, num_past, num_past, 3)
        coord_diff = coord.unsqueeze(1) - coord.unsqueeze(2)

        ### (b*n_node, num_past, num_past)
        if self.norm_diff:
            coord_diff = F.normalize(coord_diff, p=2, dim=-1)

        ## (b*n_node, num_past, 1, d)
        q = self.q_mlp(h).unsqueeze(2)
        ## (b*n_node, 1, num_past, d)
        k = self.k_mlp(h).unsqueeze(1)
        v = self.v_mlp(h).unsqueeze(1)

        ### (b*n_node, num_past, num_past)
        scores = torch.sum(q * k, dim=-1)

        if self.with_mask:
            ### (1, num_past, num_past)
            mask = subsequent_mask(scores.shape[1]).unsqueeze(0).to(h.device)
            scores = scores.masked_fill(~mask, -1e9)

        ### (b*n_node, num_past, num_past)
        tempe = 1.
        alpha = torch.softmax(scores / tempe, dim=-1)

        h_agg = torch.sum(alpha.unsqueeze(-1) * v, axis=2)

        trans = coord_diff * alpha.unsqueeze(-1) * self.coord_mlp(v)
        x_agg = torch.sum(trans, dim=2)

        coord = coord + x_agg

        # h = h_agg
        if self.recurrent:
            h = h + h_agg
        else:
            h = h_agg

        return h, coord


def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


##X


def unsorted_segment_sum_X(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1), data.size(2))
    segment_ids = segment_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, data.size(1), data.size(2))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean_X(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    result_shape = (num_segments, data.size(1), data.size(2))
    segment_ids = segment_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, data.size(1), data.size(2))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL_X(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL_X, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 4 * 4

        self.adj_mlp = nn.Sequential(
            nn.Linear(edges_in_d, 1),
            act_fn,
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        self.coord_mlp_pos = nn.Sequential(
            nn.Linear(edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 4))

        layer = nn.Linear(hidden_nf, 4, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)

        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
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
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat).unsqueeze(-1)  # * self.adj_mlp(edge_attr)#**
        # trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean_X(trans, row, num_segments=coord.size(0))

        f = agg * self.coords_weight
        coord_ = coord + f
        return coord_, f

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        # radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_diff_pro = torch.einsum("ijt,ikt->ijk", coord_diff, coord_diff).reshape(coord_diff.shape[0], -1)
        # radial = coord_diff_pro / torch.sum(coord_diff_pro**2, dim=-1, keepdim=True)
        radial = F.normalize(coord_diff_pro, dim=-1, p=2)

        return radial, coord_diff

    def egnn_mp(self, h, coord, edge_index, edge_attr, Fs=None):
        row, col = edge_index

        # (n_edge, n_channel**2)  (n_edge, n_channel, 3)
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        coord, _ = self.coord_model(coord, edge_index, coord_diff, edge_feat)  #

        h, _ = self.node_model(h, edge_index, edge_feat, node_attr=Fs)

        return [h, coord]

    def forward(self, h, coord, edge_index, edge_attr, Fs=None):
        ### because of memory limitation, we choose to operate loops on the dimention of T, instead of adding an extra dimension to the tensor.
        ### Doing this increases the running time, that is a trade-off between time and memory.
        res = list(zip(*[self.egnn_mp(h[i], coord[i], edge_index, edge_attr, Fs) for i in range(h.shape[0])]))
        hs, coords = torch.stack(res[0]), torch.stack(res[1])
        return hs, coords


class E_GCL_AT_X(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False,
                 with_mask=False):
        super(E_GCL_AT_X, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.with_mask = with_mask
        # edge_coords_nf = 4*4
        edge_coords_nf = 0

        self.q_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.k_mlp = nn.Sequential(
            nn.Linear(edge_coords_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.v_mlp = nn.Sequential(
            nn.Linear(edge_coords_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        layer = nn.Linear(hidden_nf, 4, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        # if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)

    def forward(self, h, coord):
        ### (num_past, num_past, b*n_node, 4, 3)
        coord_diff = coord[:, None, ...] - coord[None, ...]

        ### (num_past, num_past, b*n_node, 16)
        coord_diff_pro = torch.einsum("ijlkt,ijlst->ijlks", coord_diff, coord_diff).reshape(coord_diff.shape[0],
                                                                                            coord_diff.shape[1],
                                                                                            coord_diff.shape[2], -1)
        # radial = coord_diff_pro / (torch.sum(coord_diff_pro**2, dim=-1, keepdim=True)+1e-6)

        q = self.q_mlp(h).unsqueeze(1)
        k = self.k_mlp(h).unsqueeze(0)
        v = self.v_mlp(h).unsqueeze(0)
        scores = torch.sum(q * k, dim=-1)

        if self.with_mask:
            ### (num_past, num_past, 1)
            mask = subsequent_mask(scores.shape[0]).unsqueeze(-1).to(h.device)
            scores = scores.masked_fill(~mask, -1e9)

        ### (num_past, num_past, b*n_node)
        alpha = torch.softmax(scores, dim=1)

        # print(alpha[:, :, 0])

        h_agg = torch.sum(alpha.unsqueeze(-1) * v, axis=1)

        x_agg = torch.sum(coord_diff * alpha.unsqueeze(-1).unsqueeze(-1) * self.coord_mlp(v).unsqueeze(-1), axis=1)

        coord = coord + x_agg
        if self.recurrent:
            h = h + h_agg

        return h, coord