from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


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

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=0, act_fn=nn.ReLU(), bias=True, attention=False, t_eq=False, recurrent=True):
        super(GCL, self).__init__()
        self.attention = attention
        self.t_eq=t_eq
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

        #if recurrent:
            #self.gru = nn.GRUCell(hidden_nf, hidden_nf)


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
            #out = self.gru(out, h)
        return out


class GCL4E(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(),
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(GCL4E, self).__init__()
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
        
        self.var_mlp = nn.Sequential(
            nn.Linear(6, 3, bias=True),
            act_fn,
            nn.Linear(3, 3, bias=True))

        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
        

    def edge_model(self, h, edge_index, radial, edge_attr, Fs):
        row, col = edge_index
        source, target = h[row], h[col]
        # print(source.shape)
        # print(target.shape)
        # print(radial.shape)
        # print(edge_attr.shape)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=-1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=-1)

        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        return out

    def var_model(self, edge_index, coord):
        row, col = edge_index
        edge_in = torch.cat([coord[row], coord[col]], dim=2)
        var = self.var_mlp(edge_in)
        agg = unsorted_segment_sum(var, row, num_segments=3100)
        # print(var.shape)
        # exit()
        return agg


    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        # print(agg.shape)
        # exit()
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=-1)
        else:
            agg = torch.cat([x, agg], dim=-1)


        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, Fs):
        row, col = edge_index
        
        trans = coord_diff * self.coord_mlp(edge_feat) #* self.adj_mlp(edge_attr)#**
        
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train

        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        # self.coords_weight = 0.5
        f=agg*self.coords_weight
        coord_ = coord + f
        # coord_ = f
        return coord_, f

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, dim=-1, keepdim=True)

        if self.norm_diff:
            coord_diff = F.normalize(coord_diff, p=2, dim=-1)

        return radial, coord_diff


    def forward(self, h, coord, edge_index, edge_attr, Fs=None):
        radial, coord_diff = self.coord2radial(edge_index, coord)
        # var = self.var_mlp(coord)
        var = self.var_model(edge_index, coord)
        var = nn.functional.relu6(var) - 3
        
        edge_feat = self.edge_model(h, edge_index, radial, edge_attr, Fs=Fs)
        coord, _ = self.coord_model(coord, edge_index, coord_diff, edge_feat, Fs=Fs)#

        h, _ = self.node_model(h, edge_index, edge_feat, node_attr=Fs)

        return h, coord, var

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
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())


    def edge_model(self, h, edge_index, radial, edge_attr, Fs):
        row, col = edge_index
        source, target = h[row], h[col]

        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=-1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=-1)

        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))

        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=-1)
        else:
            agg = torch.cat([x, agg], dim=-1)


        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, Fs):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) #* self.adj_mlp(edge_attr)#**
        # print(self.coord_mlp(edge_feat).shape)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        # print(agg.shape)
        # exit()
        f=agg*self.coords_weight 
        coord_ = coord + f
        return coord_, f

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, dim=-1, keepdim=True)

        if self.norm_diff:
            coord_diff = F.normalize(coord_diff, p=2, dim=-1)

        return radial, coord_diff

    def forward(self, h, coord, edge_index, edge_attr, Fs=None):
        # print(coord.shape)
        radial, coord_diff = self.coord2radial(edge_index, coord)
        # print(radial.shape)
        # print(coord_diff.shape)
        # exit()
        edge_feat = self.edge_model(h, edge_index, radial, edge_attr, Fs=Fs)

        coord, _ = self.coord_model(coord, edge_index, coord_diff, edge_feat, Fs=Fs)#

        h, _ = self.node_model(h, edge_index, edge_feat, node_attr=Fs)

        return h, coord


import logging

# 设置日志配置
logging.basicConfig(filename='gamma_values.log', level=logging.INFO, format='%(asctime)s - %(message)s')



class E_GCL_AT(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False, with_mask=False):
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
            nn.Linear(edge_coords_nf+hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        
        self.v_mlp = nn.Sequential(
            nn.Linear(edge_coords_nf+hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.qs_mlp = nn.Sequential(
            nn.Linear(6, 16, bias=False),
            act_fn,
            nn.Linear(16, 6, bias=False),
            act_fn)

        self.ks_mlp = nn.Sequential(
            nn.Linear(6, 16, bias=False),
            act_fn,
            nn.Linear(16, 6, bias=False),
            act_fn)

        self.vs_mlp = nn.Sequential(
            nn.Linear(6, 16, bias=False),
            act_fn,
            nn.Linear(16, 6, bias=False),
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
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        coord_mlp_s = []
        coord_mlp_s.append(nn.Linear(6, 16))
        coord_mlp_s.append(act_fn)
        coord_mlp_s.append(nn.Linear(16, 1))
        self.coord_mlp_s = nn.Sequential(*coord_mlp_s)

        self.gamma_mlp = nn.Sequential(
            nn.Linear(hidden_nf + 6, 1),
            nn.Sigmoid()
        )

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())



    def forward(self, h, coord, stft):
        """
            h: (bs*n_node, num_past, d)
            coord: (bs*n_node, num_past, coord_dim)
            stft: (bs*n_node, coord_dim, num_past, frequency_bins)
        """
        ### (b*n_node, num_past, num_past, 3)
        coord_diff = coord.unsqueeze(1) - coord.unsqueeze(2)

        ### (b*n_node, num_past, num_past)
        if self.norm_diff:    
            coord_diff = F.normalize(coord_diff, p=2, dim=-1)

        ## (b*n_node, num_past, 1, d)
        q = self.q_mlp(h).unsqueeze(2)
        ## (b*n_node, 1, num_past, d) for both k and v
        k = self.k_mlp(h).unsqueeze(1)
        v = self.v_mlp(h).unsqueeze(1)

        # [128, 2, 8, 6]

        # 以下三个线性层，进去的dim=6， bias=False
        qs = self.qs_mlp(stft)
        ks = self.ks_mlp(stft)
        vs = self.vs_mlp(stft)

        ### (b*n_node, num_past, num_past)
        scores = torch.sum(q * k, dim=-1)

        scores_s = torch.matmul(qs, ks.transpose(-1, -2))
        # [128, 2, 8, 8]
        if self.with_mask:
            ### (1, num_past, num_past)
            mask = subsequent_mask(scores.shape[1]).unsqueeze(0).to(h.device)
            scores = scores.masked_fill(~mask, -1e9)

        ### (b*n_node, num_past, num_past)
        tempe = 1.
        alpha = torch.softmax(scores / tempe, dim=-1)
        # [128, 8, 8]

        alpha_s = torch.softmax(scores_s / tempe, dim=-1)
        # [128, 2, 8, 8]

        h_agg = torch.sum(alpha.unsqueeze(-1)*v, axis=2)
        # [128, 8, 64]

        h_agg_s = torch.matmul(alpha_s, vs)
        # [128, 2, 8, 6]

        h_agg_s_magnitude = torch.norm(h_agg_s.permute(0, 3, 2, 1), dim=-1).permute(0, 2, 1)
        # [128, 8, 6]

        # h_agg_s 这东西是个向量 而不是标量 s 向量 所以可能不能直接做这个变换我
        # attention, v -> s 的模长
        # trans_s
        # x_agg_s, x_agg_h -> 
        #  ,dim=p  nn.linear(p, 1) = alpha 
        # coord = coord + (1 - alpha) x_agg + (alpha) x_agg_s

        trans = coord_diff*alpha.unsqueeze(-1)*self.coord_mlp(v)
        # torch.Size([128, 8, 8, 2])
        # torch.Size([128, 8, 8, 1])
        # torch.Size([128, 1, 8, 1])

        # 计算模长
        vs_magnitude = torch.norm(vs.permute(0, 3, 2, 1), dim=-1, keepdim=True).permute(0, 3, 2, 1)
        # torch.Size([128, 2, 8, 6]) -> torch.Size([128, 1, 8, 6])

        alpha_s_magnitude = torch.norm(alpha_s.permute(0, 3, 2, 1), dim=-1, keepdim=True)
        # torch.Size([128, 2, 8, 8]) -> torch.Size([128, 8, 8, 1])

        trans_s = coord_diff*alpha_s_magnitude*self.coord_mlp_s(vs_magnitude)
        # torch.Size([128, 8, 8, 2])
        # torch.Size([128, 8, 8, 1])
        # torch.Size([128, 1, 8, 1])

        x_agg=torch.sum(trans, dim=2)
        # [128, 8, 2]
        x_agg_s = torch.sum(trans_s, dim=2)
        # [128, 8, 2]

        gamma_input = torch.concat((h, h_agg_s_magnitude), dim=-1)

        gamma = self.gamma_mlp(gamma_input)
        # [128, 8, 1]
        gamma_mean_per_128 = torch.mean(gamma, dim=1)  # 计算每行 8 个元素的平均值，结果形状是 [128, 1]

        # 然后计算所有 128 个的平均值
        gamma_mean_all = torch.mean(gamma_mean_per_128)  # 对 128 个平均值进行平均，结果是一个单一的数值

        # logging.info(f'gamma mean value: {gamma_mean_all}')
        # x_agg, x_agg_s,
        # concat(|s|+h), -> nn.linear(p, 1) = alpha 
        # 1. coord = coord + (1 - alpha) x_agg + (alpha) x_agg_s
        # 2. coord = coord + 0.5 * x_agg + 0.5 * x_agg_s

        coord = coord + gamma * x_agg + (1 - gamma) * x_agg_s
        
        # h = h_agg
        if self.recurrent:
            h = h + h_agg
            stft = stft + h_agg_s
        else:
            h = h_agg


        return h, coord, stft




def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
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
    result_shape = (num_segments, ) + expand_dims
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
    result_shape = (num_segments, data.size(1),data.size(2))
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
    result_shape = (num_segments, data.size(1),data.size(2))
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
        edge_coords_nf = 4*4

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
            self.coords_range = nn.Parameter(torch.ones(1))*3
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
        trans = coord_diff * self.coord_mlp(edge_feat).unsqueeze(-1) #* self.adj_mlp(edge_attr)#**
        # trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean_X(trans, row, num_segments=coord.size(0))

        f = agg*self.coords_weight
        coord_ = coord + f
        return coord_, f

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        #radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_diff_pro = torch.einsum("ijt,ikt->ijk", coord_diff,coord_diff).reshape(coord_diff.shape[0],-1)
        # radial = coord_diff_pro / torch.sum(coord_diff_pro**2, dim=-1, keepdim=True)
        radial = F.normalize(coord_diff_pro, dim=-1, p=2)

        return radial, coord_diff

    
    def egnn_mp(self, h, coord, edge_index, edge_attr, Fs=None):
        row, col = edge_index

        # (n_edge, n_channel**2)  (n_edge, n_channel, 3)
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        coord, _ = self.coord_model(coord, edge_index, coord_diff, edge_feat)#

        h, _ = self.node_model(h, edge_index, edge_feat,node_attr=Fs)

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
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False, with_mask=False):
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
            nn.Linear(edge_coords_nf+hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        
        self.v_mlp = nn.Sequential(
            nn.Linear(edge_coords_nf+hidden_nf, hidden_nf),
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
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)



    def forward(self, h, coord):
        ### (num_past, num_past, b*n_node, 4, 3)
        coord_diff=coord[:,None,...] - coord[None,...]

        ### (num_past, num_past, b*n_node, 16)
        coord_diff_pro = torch.einsum("ijlkt,ijlst->ijlks", coord_diff,coord_diff).reshape(coord_diff.shape[0],coord_diff.shape[1],coord_diff.shape[2],-1)
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
        alpha=torch.softmax(scores, dim=1)

        # print(alpha[:, :, 0])

        h_agg=torch.sum(alpha.unsqueeze(-1)*v,axis=1)

        x_agg=torch.sum(coord_diff*alpha.unsqueeze(-1).unsqueeze(-1)*self.coord_mlp(v).unsqueeze(-1),axis=1)

        coord = coord + x_agg
        if self.recurrent:
            h = h + h_agg

        return h, coord





class GMNL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(GMNL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        # edge_coords_nf = 8*8
        edge_coords_nf = 4*4

        
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

        layer = nn.Linear(hidden_nf, 4, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)

        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
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
        trans = coord_diff * self.coord_mlp(edge_feat).unsqueeze(-1) #* self.adj_mlp(edge_attr)#**
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean_X(trans, row, num_segments=coord.size(0))

        f = agg*self.coords_weight
        coord_ = coord + f
        return coord_, f

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]

        #radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        coord_diff_pro = torch.einsum("ijt,ikt->ijk", coord_diff,coord_diff).reshape(coord_diff.shape[0],-1)
        
        # radial = coord_diff_pro / torch.sum(coord_diff_pro**2, dim=-1, keepdim=True)
        radial = F.normalize(coord_diff_pro, dim=-1, p=2)

        return radial, coord_diff
    
    def egnn_mp(self, h, x, edge_index, edge_attr, Fs):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, x)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        x, _ = self.coord_model(x, edge_index, coord_diff, edge_feat)#

        h, _ = self.node_model(h, edge_index, edge_feat,node_attr=Fs)

        return h, x


    def forward(self, h, x, edge_index, edge_attr, Fs= None):
        hs=torch.stack([self.egnn_mp(h[i], x[i], edge_index, edge_attr, Fs)[0] for i in range(h.shape[0])])
        coords=torch.stack([self.egnn_mp(h[i], x[i], edge_index, edge_attr, Fs)[1] for i in range(h.shape[0])])

        return hs, coords
    

class GMN_Layer(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(),
                 recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(GMNL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        # edge_coords_nf = 8*8
        edge_coords_nf = 0

        
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

        layer = nn.Linear(hidden_nf, 4, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)

        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
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
        trans = coord_diff * self.coord_mlp(edge_feat).unsqueeze(-1) #* self.adj_mlp(edge_attr)#**
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean_X(trans, row, num_segments=coord.size(0))

        f = agg*self.coords_weight
        coord_ = coord + f
        return coord_, f

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]

        #radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        coord_diff_pro = torch.einsum("ijt,ikt->ijk", coord_diff,coord_diff).reshape(coord_diff.shape[0],-1)
        
        # radial = coord_diff_pro / torch.sum(coord_diff_pro**2, dim=-1, keepdim=True)
        radial = F.normalize(coord_diff_pro, dim=-1, p=2)

        return radial, coord_diff
    
    def egnn_mp(self, h, x, edge_index, edge_attr, Fs):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, x)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        x, _ = self.coord_model(x, edge_index, coord_diff, edge_feat)#

        h, _ = self.node_model(h, edge_index, edge_feat,node_attr=Fs)

        return h, x


    def forward(self, h, x, edge_index, edge_attr, Fs= None):
        hs=torch.stack([self.egnn_mp(h[i], x[i], edge_index, edge_attr, Fs)[0] for i in range(h.shape[0])])
        coords=torch.stack([self.egnn_mp(h[i], x[i], edge_index, edge_attr, Fs)[1] for i in range(h.shape[0])])

        return hs, coords