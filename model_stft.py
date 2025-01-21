# from tkinter.filedialog import SaveAs
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from models.gcl import E_GCL_AT, E_GCL, GCL, GMNL
# from models.gcl import GMN_Layer as GMNL
from models.layer import AGLTSA
from transformer.Models import Encoder
from einops import rearrange


#Non-equivariant STAG
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        # print(self.pe.shape)
        # exit()
        x = x + self.pe[:x.size(0)].unsqueeze(1)
        return self.dropout(x)

  
class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(SpatialBlock, self).__init__()

        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels, out_channels))

        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv1, stdv1)


    def forward(self, X, A_hat):
        lfs1 = torch.einsum("ij,kjlm->kilm", [A_hat, X])
        t1 = F.relu(torch.matmul(lfs1, self.Theta1))

        return self.batch_norm(t1)


class STAG(nn.Module):

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, out_dim=3):
        super(STAG, self).__init__()
        self.spatial = SpatialBlock(in_channels=num_features, out_channels=8,num_nodes=num_nodes)

        self.encoder = Encoder(n_layers=2, n_head=4, d_k=2, d_v=2,d_model=8,
                                 d_inner=12,  dropout=0.1, n_position=num_timesteps_input, scale_emb=False)
        
        self.theta= nn.Parameter(torch.FloatTensor(num_timesteps_input * 8, num_timesteps_output*out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)

    def forward(self, A, X):

        out1 = self.spatial(X, A)#[N, 245, 36, 8]
        out2 = self.encoder(src_seq=out1.reshape(-1,out1.shape[2],out1.shape[3]), src_mask=None, return_attns=False)[0]
        out3=torch.matmul(out2.reshape(out2.shape[0],-1), self.theta)
        return out3


class EGNN(nn.Module):
    def __init__(self,num_past, num_future, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        self.num_past=num_past
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=0,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight))

        
        self.reset_parameters()
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)



    def forward(self, h, x, edges, edge_attr):
        print(x.shape)
        exit()
        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding
        
        # h = self.PosEmbedding(h)

        permute = lambda x: x.permute(1, 0, 2)
        h, x = map(permute, [h, x])
        edge_attr = edge_attr.unsqueeze(1).repeat(1,h.shape[1],1)
        for i in range(0, self.n_layers):
            h, x = self._modules["gcl_%d" % i](h, x, edges, edge_attr, None)


        x = permute(x)
        ### only one frame
        if x.shape[0]==1:
            x_hat=x.squeeze(0)
        else:
            x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)
        # print(torch.softmax(self.theta,dim=1))
        # x_hat=torch.einsum("ij,jkt->ikt", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]
        return x_hat



# class STEGNN1(nn.Module):
#     def __init__(self,num_past, num_future, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0):
#         super(STEGNN1, self).__init__()
#         self.hidden_nf = hidden_nf
#         self.device = device
#         self.n_layers = n_layers
#         self.num_past=num_past
#         self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
#         self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
#         self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
#         self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))
#         for i in range(0, n_layers):
#             # self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=0,
#                                                 # act_fn=act_fn, recurrent=True, coords_weight=coords_weight))
#             self.add_module("gcl_%d" % i, GCL4E(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=0,
#                                                 act_fn=act_fn, recurrent=True, coords_weight=coords_weight))
#
#         self.reset_parameters()
#         self.num_particles = 20
#         self.weight_update = nn.Linear(6, 1)
#         # self.weight = nn.Parameter(1.0/self.num_particles*torch.ones((self.num_past, self.num_particles)))
#         # self.weight = nn.Parameter(1.0/self.num_particles*torch.ones((x.shape[0], self.num_particles), device=self.device))
#         self.resamp_alpha = 0.5
#         self.to(self.device)
#
#
#     def reset_parameters(self):
#         self.theta.data.uniform_(-1, 1)
#
#
#     def reparameterize(self, mu, var):
#         """
#         Reparameterization trick
#
#         :param mu: mean
#         :param var: variance
#         :return: new samples from the Gaussian distribution
#         """
#         std = torch.nn.functional.softplus(var)
#         if torch.cuda.is_available():
#             eps = torch.cuda.FloatTensor(std.shape).normal_()
#         else:
#             eps = torch.FloatTensor(std.shape).normal_()
#
#         return mu + eps * std
#
#
#     def resampling(self, particles, prob):
#         """
#         The implementation of soft-resampling. We implement soft-resampling in a batch-manner.
#
#         :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
#                         each tensor has a shape: [num_particles * batch_size, h_dim]
#         :param prob: weights for particles in the log space. Each tensor has a shape: [num_particles * batch_size, 1]
#         :return: resampled particles and weights according to soft-resampling scheme.
#         """
#         # particles: [batch, num_particle, 3]
#         # weight: [batch, num_particle]
#         resamp_prob = self.resamp_alpha * torch.exp(prob) + (1 -
#                                                              self.resamp_alpha) * 1 / self.num_particles
#         # resamp_prob = resamp_prob.view(self.num_particles, -1)
#         # indices = torch.multinomial(resamp_prob.transpose(0, 1),
#                                     # num_samples=self.num_particles, replacement=True)
#         # print(resamp_prob)
#         indices = torch.multinomial(resamp_prob,
#                                     num_samples=self.num_particles, replacement=True)
#         batch_size = indices.size(0)
#         indices = indices.transpose(1, 0).contiguous()
#         offset = torch.arange(batch_size).type(torch.LongTensor).unsqueeze(0)
#         # print(offset)
#         # exit()
#         if torch.cuda.is_available():
#             offset = offset.cuda()
#         indices = offset + indices * batch_size
#         flatten_indices = indices.view(-1, 1).squeeze()
#
#         particles_new = particles.view(batch_size * self.num_particles, 3)[flatten_indices]
#         particles_new = particles_new.view(batch_size, self.num_particles, 3)
#         prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
#         prob_new = prob_new / (self.resamp_alpha * prob_new + (1 -
#                                                                self.resamp_alpha) / self.num_particles)
#         prob_new = torch.log(prob_new).view(batch_size, self.num_particles)
#         prob_new = prob_new - torch.logsumexp(prob_new, dim=1, keepdim=True)
#         # prob_new = prob_new.view(-1, 1)
#
#         return particles_new, prob_new
#
#
#
#
#     def forward(self, h, x, edges, edge_attr):
#         h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
#         time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
#         h = h + time_embedding
#
#         # h = self.PosEmbedding(h)
#
#         permute = lambda x: x.permute(1, 0, 2)
#         h, x = map(permute, [h, x])
#         # edge_attr = edge_attr.unsqueeze(1).repeat(1,h.shape[1],1)
#         edge_attr = edge_attr.unsqueeze(1).repeat(1,self.num_particles,1)
#         # print(edge_attr)
#         # exit()
#
#         # coord_expanded = x.unsqueeze(2).repeat(1, 1, self.num_particles, 1)
#         # print(x.shape)
#         coord_expanded = x[:,0].unsqueeze(1).repeat(1, self.num_particles, 1)
#
#         h = h.unsqueeze(2).repeat(1, 1, self.num_particles, 1)
#         # edge_attr = edge_attr.unsqueeze(2).repeat(1, 1,self.num_particles,1)
#         noise = torch.randn_like(coord_expanded)
#         coord_noisy = coord_expanded + noise * 0
#         # print(coord_noisy.shape)
#         # self.weight = nn.Parameter(1.0/self.num_particles*torch.ones((x.shape[0],self.num_past, self.num_particles), device=self.device, requires_grad=True))
#         self.log_weight = 1.0/self.num_particles*torch.ones((x.shape[0], self.num_particles), device=self.device)
#         # self.log_weight = torch.ones((x.shape[0], self.num_particles), device=self.device) * np.log(1/self.num_particles)
#
#         # x : [3100, 10, 3]
#         # coord_noisy : [3100, 10, 20, 3]
#         # h: [3100, 10, 20, 16]
#         coord_k = coord_noisy
#         # for i in range(0, self.n_layers):
#             # h, x, v = self._modules["gcl_%d" % i](h, x, edges, edge_attr, None)
#
#         for k in range(0, self.num_past):
#             h_k = h[:, k]
#             for i in range(0, self.n_layers):
#                 h_k, coord_k, var = self._modules["gcl_%d" % i](h_k, coord_k, edges, edge_attr, None)
#                 # coord_k = self.reparameterize(coord_k, var)
#                 # coord_k = var
#             if k != self.num_past - 1:
#                 # print(coord_k.shape)
#                 # exit()
#                 logpdf = self.weight_update(torch.cat((coord_k, x[:, k+1].unsqueeze(1).repeat(1, self.num_particles, 1)),dim=2))
#                 #
#                 # logpdf = -0.5 * (coord_k - x[:, k+1].unsqueeze(1).repeat(1, self.num_particles, 1)) ** 2
#                 # logpdf = torch.sum(logpdf, dim=-1)
#                 # likelihood: [3100, 20, 3]
#                 self.log_weight = self.log_weight + logpdf.squeeze()
#                 self.log_weight = nn.functional.relu6(self.log_weight) - 3
#                 self.log_weight = self.log_weight - torch.logsumexp(self.log_weight, dim=1, keepdim=True)
#                 # x[:, k] = torch.einsum("ijk,ij->ik", coord_k, torch.exp(self.log_weight) + 1e-2)
#                 # print(coord_k.shape)
#                 # print(self.log_weight.shape)
#                 coord_k, self.log_weight = self.resampling(coord_k, self.log_weight)
#                 # print(self.log_weight)
#                 # print(torch.exp(self.log_weight))
#                 x[:, k] = torch.einsum("ijk,ij->ik", coord_k, torch.exp(self.log_weight))
#                 # x[:, k] = torch.einsum("ijk,ij->ik", coord_k, self.log_weight)
#                 # x[:, k] = torch.mean(coord_k, dim=1)
#
#         x = permute(x)
#         ### only one frame
#         if x.shape[0]==1:
#             x_hat=x.squeeze(0)
#         else:
#             x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)
#         # print(torch.softmax(self.theta,dim=1))
#         return x_hat


class STEGNN(nn.Module):
    def __init__(self,num_past, num_future, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0):
        super(STEGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.num_past=num_past
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))
        for i in range(0, n_layers):
            # self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=0,
                                                # act_fn=act_fn, recurrent=True, coords_weight=coords_weight))
            self.add_module("gcl_%d" % i, GCL4E(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=0,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight))
        
        self.reset_parameters()
        self.num_particles = 20
        self.weight_update = nn.Linear(1, 1)
        # self.weight_update = nn.Linear(6, 1)
        # self.weight = nn.Parameter(1.0/self.num_particles*torch.ones((self.num_past, self.num_particles)))
        # self.weight = nn.Parameter(1.0/self.num_particles*torch.ones((x.shape[0], self.num_particles), device=self.device))
        self.resamp_alpha = 0.5
        self.to(self.device)


    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)


    def reparameterize(self, mu, var):
        """
        Reparameterization trick

        :param mu: mean
        :param var: variance
        :return: new samples from the Gaussian distribution
        """
        std = torch.nn.functional.softplus(var)
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.shape).normal_()
        else:
            eps = torch.FloatTensor(std.shape).normal_()

        return mu + eps * std


    def resampling(self, particles, prob):
        """
        The implementation of soft-resampling. We implement soft-resampling in a batch-manner.

        :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
                        each tensor has a shape: [num_particles * batch_size, h_dim]
        :param prob: weights for particles in the log space. Each tensor has a shape: [num_particles * batch_size, 1]
        :return: resampled particles and weights according to soft-resampling scheme.
        """
        # particles: [batch, num_particle, 3]
        # weight: [batch, num_particle]
        resamp_prob = self.resamp_alpha * torch.exp(prob) + (1 -
                                                             self.resamp_alpha) * 1 / self.num_particles
        indices = torch.multinomial(resamp_prob,
                                    num_samples=self.num_particles, replacement=True)
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous()
        offset = torch.arange(batch_size).type(torch.LongTensor).unsqueeze(0)
        # print(offset)
        # exit()
        if torch.cuda.is_available():
            offset = offset.cuda()
        indices = offset + indices * batch_size
        flatten_indices = indices.view(-1, 1).squeeze()

        particles_new = particles.view(batch_size * self.num_particles, 3)[flatten_indices]
        particles_new = particles_new.view(batch_size, self.num_particles, 3)
        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 -
                                                               self.resamp_alpha) / self.num_particles)
        prob_new = torch.log(prob_new).view(batch_size, self.num_particles)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=1, keepdim=True)
        # prob_new = prob_new.view(-1, 1)

        return particles_new, prob_new




    def forward(self, h, x, edges, edge_attr):
        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding

        # h = self.PosEmbedding(h)

        permute = lambda x: x.permute(1, 0, 2)
        h, x = map(permute, [h, x])
        # edge_attr = edge_attr.unsqueeze(1).repeat(1,h.shape[1],1)
        edge_attr = edge_attr.unsqueeze(1).repeat(1,self.num_particles,1)
        # print(edge_attr)
        # exit()
        
        # coord_expanded = x.unsqueeze(2).repeat(1, 1, self.num_particles, 1)
        # print(x.shape)
        coord_expanded = x[:,0].unsqueeze(1).repeat(1, self.num_particles, 1)
        
        h = h.unsqueeze(2).repeat(1, 1, self.num_particles, 1)
        # edge_attr = edge_attr.unsqueeze(2).repeat(1, 1,self.num_particles,1)
        noise = torch.randn_like(coord_expanded)
        coord_noisy = coord_expanded + noise * 0.0
        # print(coord_noisy.shape)
        # self.weight = nn.Parameter(1.0/self.num_particles*torch.ones((x.shape[0],self.num_past, self.num_particles), device=self.device, requires_grad=True))
        self.log_weight = 1.0/self.num_particles*torch.ones((x.shape[0], self.num_particles), device=self.device)
        # self.log_weight = torch.ones((x.shape[0], self.num_particles), device=self.device) * np.log(1/self.num_particles)
        
        # x : [3100, 10, 3]
        # coord_noisy : [3100, 10, 20, 3]
        # h: [3100, 10, 20, 16]
        coord_k = coord_noisy
        # for i in range(0, self.n_layers):
            # h, x, v = self._modules["gcl_%d" % i](h, x, edges, edge_attr, None)

        for k in range(0, self.num_past):
            h_k = h[:, k]
            for i in range(0, self.n_layers):
                h_k, coord_k, var = self._modules["gcl_%d" % i](h_k, coord_k, edges, edge_attr, None)
                coord_k = self.reparameterize(coord_k, var)
                # coord_k = var
            if k != self.num_past - 1:
                # print(coord_k.shape)
                # exit()
                # logpdf = self.weight_update(torch.cat((coord_k, x[:, k+1].unsqueeze(1).repeat(1, self.num_particles, 1)),dim=2))
                logpdf = self.weight_update(torch.sum((coord_k - x[:, k+1].unsqueeze(1).repeat(1, self.num_particles, 1)),dim=2, keepdim=True))

                # logpdf = -0.5 * (coord_k - x[:, k+1].unsqueeze(1).repeat(1, self.num_particles, 1)) ** 2
                # logpdf = torch.sum(logpdf, dim=-1)
                # likelihood: [3100, 20, 3]
                self.log_weight = self.log_weight + logpdf.squeeze()
                self.log_weight = nn.functional.relu6(self.log_weight) - 3
                self.log_weight = self.log_weight - torch.logsumexp(self.log_weight, dim=1, keepdim=True)
                # x[:, k] = torch.einsum("ijk,ij->ik", coord_k, torch.exp(self.log_weight) + 1e-2)
                # print(coord_k.shape)
                # print(self.log_weight.shape)
                coord_k, self.log_weight = self.resampling(coord_k, self.log_weight)
                # print(self.log_weight)
                # print(torch.exp(self.log_weight))
                x[:, k] = torch.einsum("ijk,ij->ik", coord_k, torch.exp(self.log_weight))
                # x[:, k] = torch.einsum("ijk,ij->ik", coord_k, self.log_weight)
                # x[:, k] = torch.mean(coord_k, dim=1)

        x = permute(x)
        ### only one frame
        # if x.shape[0]==1:
        #     x_hat=x.squeeze(0)
        # else:
        #     x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)
        # print(torch.softmax(self.theta,dim=1))
        x_hat=torch.einsum("ij,jkt->ikt", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]
        # x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)

        return x_hat


class ESTAG(nn.Module):
    def __init__(self,  num_past, num_future, in_node_nf, in_edge_nf, hidden_nf, fft, eat, device, n_layers, n_nodes, nodes_att_dim=0,
                act_fn=nn.SiLU(), coords_weight=1.0, with_mask=False, tempo=True, filter=True):
        super(ESTAG, self).__init__()
        self.hidden_nf = hidden_nf
        self.fft = fft
        self.eat = eat
        self.device = device
        # a half for ESM, another half for ETM
        self.n_layers = int(n_layers / 2)
        self.n_nodes=n_nodes
        self.num_past = num_past
        self.tempo = tempo
        self.filter = filter
        
        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)

        for i in range(self.n_layers):
            self.add_module("egcl_%d" % (i*2+1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=nodes_att_dim,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, norm_diff=True, clamp=True))
            if self.eat:
                self.add_module("egcl_at_%d" % (i*2+2), E_GCL_AT(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, with_mask=with_mask))
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))

        self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        self.reset_parameters()
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)
        #O init
        self.theta.data*=0
        
        
    
    def FFT(self, h, x, n_nodes, edges):
        x_ = rearrange(x, 't (b n) d -> t b n d', n=n_nodes)
        x_bar = torch.mean(x_, dim=-2, keepdim=True)
        x_norm = x_ - x_bar
        # x_norm = x_
        x_norm = rearrange(x_norm, 't b n d -> (b n) d t')
        
        ### (b*n_node, 3, num_past)
        # print(x_norm.shape)
        
        F = torch.fft.fftn(x_norm, dim=-1)
        if self.filter:
            attn_val = self.attn_mlp(h[1:]).squeeze(-1).transpose(0, 1)
        else:
            # (b*n_node,), broadcast
            attn_val = torch.ones(h.shape[1], device=h.device).unsqueeze(-1)

        F = F[..., 1:]
        F_i = F[edges[0]]
        F_j = F[edges[1]]
        # print(edges[0].shape)
        ## (n_egde, num_past-1)
        edge_attr = torch.abs(torch.sum(torch.conj(F_i) * F_j, dim=-2))
        
        edge_attr = edge_attr * (attn_val[edges[0]] * attn_val[edges[1]])

        edge_attr_norm = edge_attr / (torch.sum(edge_attr, dim=-1, keepdim=True)+1e-9)

        ### (b*n_node, num_past-1)
        Fs = torch.abs(torch.sum(F**2, dim=-2))
        
        Fs = Fs * attn_val

        Fs_norm = Fs / (torch.sum(Fs, dim=-1, keepdim=True)+1e-9)
        return edge_attr_norm, Fs_norm


    def forward(self, h, x, edges, edge_attr):
        """parameters
            h: (b*n_node, 1)
            x: (num_past, b*n_node, 3)
            edges: (2, n_edge)
            edge_attr: (n_edge, 3)
        """ 

        ### (num_past, b*n_node, hidden_nf)
        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
        
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding
        # h = self.PosEmbedding(h)

        Fs=None
        if self.fft:
            ### (n_edge, num_past-1), (b*n_node, num_past-1)
            # print(h.shape)
            # print(x.shape)
            edge_attr, Fs = self.FFT(h, x, self.n_nodes, edges=edges)
        

        permute = lambda x: x.permute(1, 0, 2)
        h, x = map(permute, [h, x])
        if Fs is not None: Fs = Fs.unsqueeze(1).repeat(1,h.shape[1],1)
        edge_attr = edge_attr.unsqueeze(1).repeat(1,h.shape[1],1)

        for i in range(self.n_layers):
            h, x = self._modules["egcl_%d" % (i*2+1)](h, x, edges, edge_attr, Fs)

            if self.eat:
                h, x = self._modules["egcl_at_%d" % (i*2+2)](h, x)

        x = permute(x)
        # self.tempo = False
        if self.tempo:
            x_hat=torch.einsum("ij,jkt->ikt", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]
        else:
            x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)

        return x_hat



def cal_similarity_fourier(fourier_features):
    similarity=torch.abs(torch.mm(torch.conj(fourier_features), fourier_features.t()))
    return similarity





class GNN(nn.Module):
    def __init__(self, num_future, num_past, input_dim, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=0, recurrent=False):
        super(GNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.num_past = num_past

        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf = in_edge_nf ,
                                              act_fn=act_fn, attention=attention, recurrent=recurrent))

        self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
                              act_fn,
                              nn.Linear(hidden_nf,3))
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))

        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))
        self.reset_parameters()
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)


    def forward(self, nodes, edges, edge_attr=None):
        h = self.embedding(nodes)
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding
        # h = self.PosEmbedding(h)
        
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)

        x = self.decoder(h)
        x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)
        # print(torch.softmax(self.theta,dim=1))
        return x_hat


# class Linear_comb(nn.Module):
#     def __init__(self, num_future, num_past, input_dim, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(),
#                  n_layers=4, attention=0, recurrent=False):
#         super(GNN, self).__init__()
#         self.hidden_nf = hidden_nf
#         self.device = device
#         self.n_layers = n_layers
#         self.num_past = num_past
#
#         self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
#         self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
#         for i in range(0, n_layers):
#             self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=in_edge_nf,
#                                               act_fn=act_fn, attention=attention, recurrent=recurrent))
#
#         self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
#                                      act_fn,
#                                      nn.Linear(hidden_nf, 3))
#         self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))
#
#         self.theta = nn.Parameter(torch.FloatTensor(num_future, num_past))
#         self.reset_parameters()
#         self.to(self.device)
#
#     def reset_parameters(self):
#         self.theta.data.uniform_(-1, 1)
#
#     def forward(self, nodes, edges, edge_attr=None):
#         h = self.embedding(nodes)
#         time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
#         h = h + time_embedding
#         # h = self.PosEmbedding(h)
#
#         # for i in range(0, self.n_layers):
#         #     h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
#
#         # x = self.decoder(h)
#         x_hat = torch.einsum("ij,jkt->ikt", torch.softmax(self.theta, dim=1), node).squeeze(0)
#         # print(torch.softmax(self.theta,dim=1))
#         return x_hat



class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=2):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)#100 13 10 6 -> 100 6 13 10  #100 13 8 16 #100 13 6 64 # 100 13 4 16

        ######## + -> *
        #100 64 13 9
        temp = self.conv1(X) * torch.sigmoid(self.conv2(X))

        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """

        ### (b, n, t, c)
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output, out_dim, device):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()

        self.device = device
        self.embedding = nn.Linear(num_features, 32)
        self.block1 = STGCNBlock(in_channels=32, out_channels=64,
                                 spatial_channels=32, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=32, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        

        #### 1 * 5 = 1 * (2 * 2 + 1)   (1 is kernel_size-1,    block1 -2  |  block2 -2  | last_temporal, -1)
        # self.fully = nn.Linear((num_timesteps_input - 1 * 5) * 64,
        #                        num_timesteps_output*out_dim)
        self.fully = nn.Linear((num_timesteps_input - 1 * 4) * 64,
                               num_timesteps_output*out_dim)


        self.theta= nn.Parameter(torch.FloatTensor(1, num_timesteps_output))
        self.reset_parameters()
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)
        

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        X = self.embedding(X)
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        # out3 = self.last_temporal(out2)
        out3 = out2
        # print(out2.shape)
        # print(out3.shape)
        # exit()
        # print(out3.reshape((out3.shape[0], out3.shape[1], -1)).shape)
        # print(out2.reshape((out3.shape[0], out3.shape[1], -1)).shape)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))

        # x = rearrange(out4, 'b n (t d) -> t (b n) d', d=3)
        # x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)

        # return x_hat

        return out4



class AGLSTAN(nn.Module):
    ### embed_dim is d_e
    def __init__(self, num_nodes, batch_size, input_dim, output_dim, window, num_layers, filter_size, embed_dim, cheb_k, num_future):
        super(AGLSTAN, self).__init__()
        self.num_node = num_nodes
        self.batch_size = batch_size
        ### K
        self.input_dim = input_dim
        self.num_future=num_future
        
        ### F
        self.output_dim = output_dim
        
        ### alpha
        self.window = window
        self.num_layers = num_layers
        self.filter_size = filter_size

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node,embed_dim), requires_grad=True)

        self.encoder = AGLTSA(num_nodes, input_dim, output_dim, cheb_k,
                                embed_dim, num_nodes * self.output_dim, filter_size, num_layers)

        self.end_conv = nn.Conv2d(in_channels=self.window, out_channels=num_future, padding=(2, 2), kernel_size=(5, 5), bias=True)

    def forward(self, source):
        #source: B, T_1, N, D
        #target: B, T_2, N, D

        output = self.encoder(source, self.node_embeddings)   #B, T, N, hidden

        
        output = output.view(self.batch_size, self.window, self.num_node, -1)
        output = self.end_conv(output)

        return output

class GMN(nn.Module):
    def __init__(self,num_past, num_future,  in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.LeakyReLU(0.2), n_layers=4, coords_weight=3.0):
        super(GMN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.num_past = num_past

        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))
        for i in range(0, n_layers):
            self.add_module("gmnl_%d" % i, GMNL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight))
        
        self.reset_parameters()
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)



    def forward(self, h, x, edges, edge_attr):
        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding

        for i in range(0, self.n_layers):
            h, x = self._modules["gmnl_%d" % i](h, x, edges,edge_attr=edge_attr)
        
        if x.shape[0]==1:
            x_hat=x.squeeze(0)
        else:
            x_hat = torch.einsum("ij,jkts->ikts", torch.softmax(self.theta,dim=1), x).squeeze(0)
        return x_hat


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class ETimesNet(nn.Module):
    def __init__(self,  num_past, num_future, in_node_nf, in_edge_nf, hidden_nf, fft, eat, device, n_layers, n_nodes, nodes_att_dim=0,
                act_fn=nn.SiLU(), coords_weight=1.0, with_mask=False, tempo=True, filter=True):
        super(ETimesNet, self).__init__()
        self.hidden_nf = hidden_nf
        self.fft = fft
        self.k = 2
        self.eat = eat
        self.device = device
        self.n_layers = n_layers
        self.n_nodes=n_nodes
        self.num_past = num_past
        self.tempo = tempo
        self.filter = filter
        # self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf)
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)

        for i in range(n_layers):
            self.add_module("prior_egcl_%d" % (i*2+1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=nodes_att_dim,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, norm_diff=True, clamp=True))


        for i in range(n_layers):
            self.add_module("egcl_%d" % (i*2+1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=nodes_att_dim,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, norm_diff=True, clamp=True))
            if self.eat:
                self.add_module("egcl_at_%d" % (i*2+2), E_GCL_AT(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, with_mask=with_mask))
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))

        self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        self.conv = nn.Sequential(
            Inception_Block_V1(16, 16,
                               num_kernels=2),
            # nn.GELU(),
            # Inception_Block_V1(16, 16,
                            #    num_kernels=2)
        )
        self.conv_x = nn.Sequential(
            Inception_Block_V1(3, 3,
                               num_kernels=2),
            # nn.GELU(),
            # Inception_Block_V1(16, 16,
                            #    num_kernels=2)
        )

        self.reset_parameters()
        self.seq_len = self.num_past
        self.pred_len = 1
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)
        #O init
        self.theta.data*=0
        
        
    
    def FFT(self, h, x, n_nodes, edges, k=5):
        k = self.k
        x_ = rearrange(x, 't (b n) d -> t b n d', n=n_nodes)
        x_bar = torch.mean(x_, dim=-2, keepdim=True)
        x_norm = x_ - x_bar
        # x_norm = rearrange(x_norm, 't b n d -> (b n) d t')
        x_norm = rearrange(x_norm, 't b n d -> b n d t')
        # x_norm, t b n d
        
        ### (b*n_node, 3, num_past)
        
        F = torch.fft.rfftn(x_norm, dim=-1)
        abs_F = abs(F)
        frequency_list = torch.sqrt(torch.mean(abs(F)**2, dim=-2)).mean(0)
        frequency_list[:, 0] = 0
        
        # print(frequency_list.shape)
        # frequency_list = frequency_list.mean(0)
        _, top_list = torch.topk(frequency_list, k)
        # print(top_list)
        # print(top_list.shape)
        # top_list = top_list.detach().cpu().numpy()
        period = x_norm.shape[-1] // top_list
        shape = abs_F.shape
        weight = torch.zeros((shape[0], shape[1], shape[2], k)).to(x.device)
        # print(weight.shape)
        # exit()
        for i in range(weight.shape[1]):
            weight[:,i] = abs_F[:, i, ...,top_list[i]]
        weight = torch.sqrt((weight**2).mean(-2))
        # weight = rearrange(weight, 'b n k -> (b n)k')
        # print(period.shape)
        # print(weight.shape)
        # exit()
        # print(period)
        # print(weight)
        # print(period.shape)
        # print(weight.shape)
        # exit()
        # print(x.shape)
        # exit()
        # period = 20 * torch.ones((n_nodes, k), dtype=int).to(x.device)
        # weight = 1.0 / 100 * torch.ones((int(x.shape[1]/n_nodes), n_nodes, k)).to(x.device)
        # print(period.shape)
        # print(weight.shape)
        # exit()

        return period, weight


    def forward(self, h, x, edges, edge_attr):
        """parameters
            h: (b*n_node, 1)
            x: (num_past, b*n_node, 3)
            edges: (2, n_edge)
            edge_attr: (n_edge, 3)
        """ 

        ### (num_past, b*n_node, hidden_nf)
        # x_c = x.clone()
        # x = torch.cat([x, x[-1].clone().unsqueeze(0)], dim=0)
        # x = x[:-1] - x[1:]
        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
        
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding
        # print(h.shape)
        # print(time_embedding.shape)
        # exit()
        # h = self.PosEmbedding(h)
        
        
        Fs=None
        if self.fft:
            ### (n_edge, num_past-1), (b*n_node, num_past-1)
            period_list, period_weight = self.FFT(h, x, self.n_nodes, edges=edges)
        

        permute = lambda x: x.permute(1, 0, 2)
        h, x = map(permute, [h, x])
        if Fs is not None: Fs = Fs.unsqueeze(1).repeat(1,h.shape[1],1)
        

        # period : [n, k]
        # weight : [b, n, k]
        # h: [bn, t, emb]
        # x: [bn, t, 3]
        h = self.predict_linear(h.permute(0, 2, 1)).permute(0, 2, 1)
        edge_attr = edge_attr.unsqueeze(1).repeat(1,h.shape[1],1)
        padding = x[:, -1,:].unsqueeze(-2).repeat(1, self.pred_len, 1)
        x = torch.cat([x, padding], dim=1)
        # exit()
        for i in range(self.n_layers):
            h, x = self._modules["prior_egcl_%d" % (i*2+1)](h, x, edges, edge_attr, Fs)

        res = []
        res_x = []
        for i in range(self.k):
            periods = period_list[:, i]
            uni_periods = torch.unique(periods)
            max_length = 0
            for period in uni_periods:
                if (self.seq_len + self.pred_len) % period != 0:
                    add_length = (((self.seq_len + self.pred_len) // period) + 1) * period
                    # print(max_length)
                    max_length = max(int(add_length), max_length)
                    # print(uni_periods)
                    # print(period)
                    # print(add_length)
                    # print(max_length)
            # if max_length != self.seq_len + self.pred_len:
            # print(h.shape)
            # print(h.shape[-2])
            # print(max_length)
            if max_length > h.shape[-2]:
                # pad_length = h.shape[-2] - (self.seq_len + self.pred_len)
                pad_length = max_length - h.shape[-2]
                # print(pad_length)
                # print(max_length)
                # print(h.shape[-2])
                # exit()
                padding = h[:, -1,:].unsqueeze(-2).repeat(1, pad_length, 1)
                h = torch.cat([h, padding], dim=1)
                # padding = x[:, -1,:].unsqueeze(-2).repeat(1, pad_length, 1)
                # x = torch.cat([x, padding], dim=1)
            # x = rearrange(x, '(b n) t d -> b n t d', n=self.n_nodes)
            h = rearrange(h, '(b n) t d -> b n t d', n=self.n_nodes)

            for period in uni_periods:
                shape_1 = (max_length // period) * period
                idx = torch.where(periods==period)[0]
                h_ = h[:, idx, :shape_1].clone()
                # x_ = x[:, idx, :shape_1].clone()
                h_ = rearrange(h_, 'b n (r p) d -> (b n) d r p', p=period)
                # x_ = rearrange(x_, 'b n (r p) d -> (b n) d r p', p=period)
                h_ = self.conv(h_)
                # x_ = self.conv_x(x_)
                h_ = rearrange(h_, 'a d n p -> a (n p) d ')
                # x_ = rearrange(x_, 'a d n p -> a (n p) d ')
                h_ = rearrange(h_, '(b n) t d -> b n t d ', n=len(idx))
                # x_ = rearrange(x_, '(b n) t d -> b n t d ', n=len(idx))
                h[:, idx, :shape_1] = h_
                # x[:, idx, :shape_1] = x_
            h = rearrange(h, 'b n t d -> (b n) t d', n=self.n_nodes)
            # x = rearrange(x, 'b n t d -> (b n) t d', n=self.n_nodes)
            res.append(h[:,:(self.seq_len + self.pred_len),:])
            # res_x.append(x[:,:(self.seq_len + self.pred_len),:])
            
        period_weight = rearrange(period_weight, 'b n k -> (b n) k')
        period_weight = F.softmax(period_weight, dim=1)
        res = torch.stack(res, dim=1)
        # print(res.shape)
        # print(period_weight.shape)
        res = torch.einsum('bktd,bk->btd', res, period_weight)
        # res_x = torch.stack(res_x, dim=1)
        # res_x = torch.einsum('bktd,bk->btd', res_x, period_weight)
        
        h = res + h[:, :(self.seq_len + self.pred_len), :]
        # x = res_x + x[:, :(self.seq_len + self.pred_len), :]
        for i in range(self.n_layers):
            h, x = self._modules["egcl_%d" % (i*2+1)](h, x, edges, edge_attr, Fs)
            if self.eat:
                h, x = self._modules["egcl_at_%d" % (i*2+2)](h, x)

        x = permute(x)
        # self.tempo = False
        # if self.tempo:
        #     x_hat=torch.einsum("ij,jkt->ikt", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]
        # else:
        #     x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)
        
        # return torch.einsum("ij,jkt->ikt", self.theta, x.unsqueeze(0)).squeeze(0)+x_c[-1]
        # return x[-1, :,:]
        x = x[:-1]
        
        x_hat=torch.einsum("ij,jkt->ikt", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]

        return x_hat


# class SpectralAttention(nn.Module):
#     def __init__(self, hidden_nf, num_channels=2):
#         super(SpectralAttention, self).__init__()
#         self.query = nn.Linear(num_channels, hidden_nf)
#         self.key = nn.Linear(num_channels, hidden_nf)
#         self.value = nn.Linear(num_channels, hidden_nf)
#         self.attn_mlp = nn.Sequential(
#             nn.Linear(hidden_nf, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, stft_sqr):
#         """
#         Parameters:
#             stft_sqr: [batch_size, num_past, num_freq_bins, num_channels] - STFT features with stacked channels.
#
#         """
#         q = self.query(stft_sqr)
#         k = self.key(stft_sqr)
#         v = self.value(stft_sqr)
#         '''
#         q torch.Size([128, 6, 5, 64])
#         k torch.Size([128, 6, 5, 64])
#         v torch.Size([128, 6, 5, 64])
#         '''
#         scores = torch.sum(q * k, dim=-1)  # [batch_size, num_freq_bins, num_frames]
#
#         alpha = torch.softmax(scores, dim=-1)  # [batch_size, num_freq_bins, num_frames]
#
#         h_agg = torch.sum((alpha.unsqueeze(-1) * v).clone(), dim=2)  # [batch_size, num_freq_bins, hidden_nf]
#
#         return h_agg

class STFT(nn.Module):
    def __init__(self,  num_past, num_future, in_node_nf, in_edge_nf, hidden_nf, fft, eat, device, n_layers, n_nodes, nodes_att_dim=0,
                act_fn=nn.SiLU(), coords_weight=1.0, with_mask=False, tempo=True, filter=True):
        super(STFT, self).__init__()
        self.hidden_nf = hidden_nf
        self.fft = fft
        self.k = 2
        self.eat = eat
        self.device = device
        self.n_layers = n_layers
        self.n_nodes=n_nodes
        self.num_past = num_past
        self.tempo = tempo
        self.filter = filter
        # self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf)
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.query = nn.Linear(6, 6)
        self.key = nn.Linear(6, 6)
        self.value = nn.Linear(6, 6)
        self.embedding1 = nn.Linear(hidden_nf, int(self.hidden_nf / 2))
        self.embedding2 = nn.Linear(1, int(self.hidden_nf / 2))

        # self.hidden_nf += 6



        for i in range(n_layers):
            self.add_module("egcl_%d" % (i*2+1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=nodes_att_dim,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, norm_diff=True, clamp=True))
            if self.eat:
                self.add_module("egcl_at_%d" % (i*2+2), E_GCL_AT(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, with_mask=with_mask))
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))

        self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 5),
                nn.Sigmoid())



        self.reset_parameters()
        self.seq_len = self.num_past
        self.pred_len = 1
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)
        #O init
        self.theta.data*=0
        
        
    



    def forward(self, h, x, edges, edge_attr):
        """parameters
            h: (b*n_node, 1)
            x: (num_past, b*n_node, 3)
            edges: (2, n_edge)
            edge_attr: (n_edge, 3)
        """
        # Calculate angel embedding
        print(h.shape)
        # [128, 8, 1])
        vel = torch.zeros_like(x)
        vel[:, 1:] = x[:, 1:] - x[:, :-1]
        vel[:, 0] = vel[:, 1]

        vel_pre = torch.zeros_like(vel)
        vel_pre[:, 1:] = vel[:, :-1]
        vel_pre[:, 0] = vel[:, 0]
        EPS = 1e-6
        vel_cosangle = torch.sum(vel_pre * vel, dim=-1) / (
                    (torch.norm(vel_pre, dim=-1) + EPS) * (torch.norm(vel, dim=-1) + EPS))

        vel_angle = torch.acos(torch.clamp(vel_cosangle, -1, 1)).unsqueeze(-1)
        # [128, 8, 1]

        h = h.permute(1, 0, 2)  # (num_past, batch_num_nodes, feature_dim)
        x = x.permute(1, 0, 2)  # (num_past, batch_num_nodes, coord_dim)



        h = self.embedding(h)
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding

        permute = lambda x: x.permute(1, 0, 2)
        h, x = map(permute, [h, x])

        h = self.embedding1(h)
        vel_angle = self.embedding2(vel_angle)
        h = torch.cat([h, vel_angle], dim=-1)
        # h: (batch_num_nodes, num_past, 64)
        # x: (batch_num_nodes, num_past, coord_dim)
        Fs=None


        # Needs to be altered as Task Changes
        n_fft = 10
        
        hop_length = 2
        # 窗长
        win_length = 4
        
        window = torch.hann_window(win_length).to(x.device)
        
        
        stft_results = []
        
        # 这里的channel是 xyz 三个坐标轴的意思， 做STFT
        for channel in range(x.shape[-1]):
            stft_result = torch.stft(
                x[:, :, channel], 
                n_fft=n_fft,
                hop_length=hop_length, 
                win_length=win_length, 
                window=window, 
                return_complex=True
            )
            stft_results.append(stft_result.abs())
        # torch.stack
        stft_tensor = torch.stack(stft_results, dim=-1)
        stft_tensor = torch.permute(stft_tensor, (0, 2, 1, 3))
        for i in range(0, h.shape[1]//hop_length):
            stft_tensor[:, i] = (stft_tensor[:, i] + stft_tensor[:, i+1]) / 2        # [128, 6, 5, 2]
        stft_tensor = stft_tensor[:, :h.shape[1]//hop_length]
        stft_tensor = stft_tensor.repeat_interleave(2, dim=1).permute(0, 3, 1, 2)
        # [128, 2, 8, 6]

        # print("stft_result.shape:", stft_results[0].shape)
        # [128, 6, 5]  6 = 10/2 + 1

        # 把xyz三个维度的频域统一，得到最后的feature
        # if x.shape[-1] == 3:
        #     stft_sqr = torch.sqrt((stft_results[0] ** 2 + stft_results[1] ** 2 + stft_results[2] ** 2) / 3)
        # else:
        #     stft_sqr = torch.sqrt((stft_results[0] ** 2 + stft_results[1] ** 2) / 2)

        # stft_sqr = stft_sqr.permute(0, 2, 1)
        # [128, 5, 6]

        # 把重合的频域特征融合在一起
        # for i in range(0, h.shape[1]//hop_length):
        #     stft_sqr[:, i] = (stft_sqr[:, i] + stft_sqr[:, i+1]) / 2
        # [128, 5, 6]


        # 把频域的特征和时域对齐
        # stft_sqr = stft_sqr[:, :h.shape[1]//hop_length]
        # [128, 4, 6]
        # Number of repeats neads to be altered based on exact task
        # stft_sqr = stft_sqr.repeat_interleave(2, dim=1)
        # [128, 8, 6]

        # 与原来的feature叠加
        # h = torch.cat((h, stft_sqr), dim=2)

        # period : [n, k]
        # weight : [b, n, k]
        edge_attr = edge_attr.unsqueeze(1).repeat(1,h.shape[1],1)

        for i in range(self.n_layers):
            h, x = self._modules["egcl_%d" % (i * 2 + 1)](h, x, edges, edge_attr, Fs)

            if self.eat:
                h, x, stft_tensor = self._modules["egcl_at_%d" % (i * 2 + 2)](h, x, stft_tensor)

        x = permute(x)
        
        x_hat=torch.einsum("ij,jkt->ikt", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]

        return x_hat




# multi-scale STFT
class MS_STFT(nn.Module):
    def __init__(self,  num_past, num_future, in_node_nf, in_edge_nf, hidden_nf, fft, eat, device, n_layers, n_nodes, nodes_att_dim=0,
                act_fn=nn.SiLU(), coords_weight=1.0, with_mask=False, tempo=True, filter=True):
        super(MS_STFT, self).__init__()
        self.hidden_nf = hidden_nf
        self.fft = fft
        self.k = 2
        self.eat = eat
        self.device = device
        self.n_layers = n_layers
        self.n_nodes=n_nodes
        self.num_past = num_past
        self.tempo = tempo
        self.filter = filter
        # self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf)
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)

        self.hidden_nf += 3
        self.hidden_nf += 6
        self.hidden_nf += 11
        # self.hidden_nf += 21
        # self.hidden_nf += 51
        for i in range(n_layers):
            self.add_module("prior_egcl_%d" % (i*2+1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=nodes_att_dim,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, norm_diff=True, clamp=True))


        for i in range(n_layers):
            self.add_module("egcl_%d" % (i*2+1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=nodes_att_dim,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, norm_diff=True, clamp=True))
            if self.eat:
                self.add_module("egcl_at_%d" % (i*2+2), E_GCL_AT(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, with_mask=with_mask))
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))

        self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())



        self.reset_parameters()
        self.seq_len = self.num_past
        self.pred_len = 1
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)
        #O init
        self.theta.data*=0
        
        
    
    def FFT(self, h, x, n_nodes, edges, k=5):
        k = self.k
        x_ = rearrange(x, 't (b n) d -> t b n d', n=n_nodes)
        x_bar = torch.mean(x_, dim=-2, keepdim=True)
        x_norm = x_ - x_bar
        # x_norm = rearrange(x_norm, 't b n d -> (b n) d t')
        x_norm = rearrange(x_norm, 't b n d -> b n d t')
        # x_norm, t b n d
        
        ### (b*n_node, 3, num_past)
        
        F = torch.fft.rfftn(x_norm, dim=-1)
        abs_F = abs(F)
        frequency_list = torch.sqrt(torch.mean(abs(F)**2, dim=-2)).mean(0)
        frequency_list[:, 0] = 0
        _, top_list = torch.topk(frequency_list, k)
        period = x_norm.shape[-1] // top_list
        shape = abs_F.shape
        weight = torch.zeros((shape[0], shape[1], shape[2], k)).to(x.device)
        for i in range(weight.shape[1]):
            weight[:,i] = abs_F[:, i, ...,top_list[i]]
        weight = torch.sqrt((weight**2).mean(-2))

        return period, weight


    def forward(self, h, x, edges, edge_attr):
        """parameters
            h: (b*n_node, 1)
            x: (num_past, b*n_node, 3)
            edges: (2, n_edge)
            edge_attr: (n_edge, 3)
        """         

        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
        
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding


        permute = lambda x: x.permute(1, 0, 2)
        h, x = map(permute, [h, x])
        
        Fs=None
        
        # win_lengths = [5, 10, 20, 30]
        # hop_lengths = [2, 5, 10, 20]
        # 得根据整个的T来选择，100
        hop_lengths = [2, 5, 10]
        
        n_fft = 10
        for hop_length in hop_lengths:
            # hop_length = win_length//2
            win_length = hop_length*2
            n_fft = win_length
            # win_length = 10
            window = torch.hann_window(win_length).to(x.device)
            stft_results = []



            for channel in range(x.shape[-1]):
                # 对每个通道进行STFT变换
                stft_result = torch.stft(
                    x[:, :, channel], 
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window, 
                    return_complex=True  # 让结果为复数形式，比较容易处理
                )
                stft_results.append(stft_result.abs())
            stft_sqr = torch.sqrt((stft_results[0]**2 + stft_results[1]**2 + stft_results[2]**2)/3)
            stft_sqr = stft_sqr.permute(0, 2, 1)
            for i in range(0, h.shape[1]//hop_length):
                stft_sqr[:, i] = (stft_sqr[:, i] + stft_sqr[:, i+1]) / 2 
            stft_sqr = stft_sqr[:, :h.shape[1]//hop_length]
            stft_sqr = stft_sqr.repeat_interleave(hop_length, dim=1)
            
            h = torch.cat((h, stft_sqr), dim=2)
        
        # print(h.shape)
        # exit()
        # period : [n, k]
        # weight : [b, n, k]
        # h: [bn, t, emb]
        # x: [bn, t, 3]
        edge_attr = edge_attr.unsqueeze(1).repeat(1,h.shape[1],1)
        # exit()
        
        
        # for i in range(self.n_layers):
            # h, x = self._modules["prior_egcl_%d" % (i*2+1)](h, x, edges, edge_attr, Fs)
        

        for i in range(self.n_layers):
            h, x = self._modules["egcl_%d" % (i*2+1)](h, x, edges, edge_attr, Fs)
            # if self.eat:
                # h, x = self._modules["egcl_at_%d" % (i*2+2)](h, x)

        x = permute(x)
        
        x_hat=torch.einsum("ij,jkt->ikt", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]

        return x_hat


class SEGNN(nn.Module):
    def __init__(self,  num_past, num_future, in_node_nf, in_edge_nf, hidden_nf, fft, eat, device, n_layers, n_nodes, nodes_att_dim=0,
                act_fn=nn.SiLU(), coords_weight=1.0, with_mask=False, tempo=True, filter=True):
        super(SEGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.fft = fft
        self.k = 2
        self.eat = eat
        self.device = device
        self.n_layers = n_layers
        self.n_nodes=n_nodes
        self.num_past = num_past
        self.tempo = tempo
        self.filter = filter
        # self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf)
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)

        self.hidden_nf += 6
        for i in range(n_layers):
            self.add_module("prior_egcl_%d" % (i*2+1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=nodes_att_dim,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, norm_diff=True, clamp=True))

        for i in range(n_layers):
            self.add_module("egcl_%d" % (i*2+1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=nodes_att_dim,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, norm_diff=True, clamp=True))
            if self.eat:
                self.add_module("egcl_at_%d" % (i*2+2), E_GCL_AT(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, with_mask=with_mask))
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))

        self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())



        self.reset_parameters()
        self.seq_len = self.num_past
        self.pred_len = 1
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)
        #O init
        self.theta.data*=0
        
        
    
    def FFT(self, h, x, n_nodes, edges, k=5):
        k = self.k
        x_ = rearrange(x, 't (b n) d -> t b n d', n=n_nodes)
        x_bar = torch.mean(x_, dim=-2, keepdim=True)
        x_norm = x_ - x_bar
        # x_norm = rearrange(x_norm, 't b n d -> (b n) d t')
        x_norm = rearrange(x_norm, 't b n d -> b n d t')
        # x_norm, t b n d
        
        ### (b*n_node, 3, num_past)
        
        F = torch.fft.rfftn(x_norm, dim=-1)
        abs_F = abs(F)
        frequency_list = torch.sqrt(torch.mean(abs(F)**2, dim=-2)).mean(0)
        frequency_list[:, 0] = 0
        _, top_list = torch.topk(frequency_list, k)
        period = x_norm.shape[-1] // top_list
        shape = abs_F.shape
        weight = torch.zeros((shape[0], shape[1], shape[2], k)).to(x.device)
        for i in range(weight.shape[1]):
            weight[:,i] = abs_F[:, i, ...,top_list[i]]
        weight = torch.sqrt((weight**2).mean(-2))

        return period, weight


    def forward(self, h, x, edges, edge_attr):
        """parameters
            h: (b*n_node, 1)
            x: (num_past, b*n_node, 3)
            edges: (2, n_edge)
            edge_attr: (n_edge, 3)
        """         

        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
        
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding


        permute = lambda x: x.permute(1, 0, 2)
        h, x = map(permute, [h, x])
        
        Fs=None
        
        n_fft = 10
        hop_length = 5
        win_length = 10
        window = torch.hann_window(win_length).to(x.device)
        
        stft_results = []
        
        for channel in range(x.shape[-1]):
            stft_result = torch.stft(
                x[:, :, channel], 
                n_fft=n_fft,
                hop_length=hop_length, 
                win_length=win_length, 
                window=window, 
                return_complex=True
            )
            stft_results.append(stft_result.abs())
        
        stft = torch.stack(stft_results, dim=3)
        stft = stft.permute(0, 2, 1, 3)
        stft = stft[:, :h.shape[1]//hop_length]
        stft = stft.repeat_interleave(5, dim=1)
        # stft = 
        # print(stft.shape)
        # # 
        stft_sqr = torch.sqrt((stft_results[0]**2 + stft_results[1]**2 + stft_results[2]**2)/3)
        stft_sqr = stft_sqr.permute(0, 2, 1)
        for i in range(0, h.shape[1]//hop_length):
            stft_sqr[:, i] = (stft_sqr[:, i] + stft_sqr[:, i+1]) / 2
        
        stft_sqr = stft_sqr[:, :h.shape[1]//hop_length]
        stft_sqr = stft_sqr.repeat_interleave(5, dim=1)
        h = torch.cat((h, stft_sqr), dim=2)
        # period : [n, k]
        # weight : [b, n, k]
        # h: [bn, t, emb]
        # x: [bn, t, 3]
        edge_attr = edge_attr.unsqueeze(1).repeat(1,h.shape[1],1)

        for i in range(self.n_layers):
            h, x = self._modules["egcl_%d" % (i*2+1)](h, x, edges, edge_attr, Fs)
            # if self.eat:
                # h, x = self._modules["egcl_at_%d" % (i*2+2)](h, x)
            # h = torch.fft.fft(h, dim=1).real + h

        x = permute(x)
        
        x_hat=torch.einsum("ij,jkt->ikt", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]

        return x_hat

