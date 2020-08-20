import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math


class HGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, n_heads, dropout=0.2, **kwargs):
        super(HGTConv, self).__init__(aggr='add', **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.n_heads = n_heads
        self.d_h = math.sqrt(self.n_heads)
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None
        self.att_score = None

        self.sub_linear = nn.ModuleList()
        self.neigh_linear = nn.ModuleList()
        self.v_linear = nn.Linear(in_dim, out_dim)
        self.norm=nn.LayerNorm(out_dim)
        self.relation_att = nn.Parameter(torch.Tensor(self.num_types, n_heads, self.d_k))
        self.relation_h_att = nn.Parameter(torch.Tensor(self.num_types, n_heads, 8))
        for t in range(num_types):
            self.sub_linear.append(nn.Linear(in_dim, out_dim))
            self.neigh_linear.append(nn.Linear(in_dim, out_dim))
        self.drop = nn.Dropout(dropout)
        self.h_sub_att = nn.Linear(8, 64)
        self.h_neigh_att = nn.Linear(8, 64)
        glorot(self.relation_att)
        glorot(self.relation_h_att)

    def forward(self, node_inp, node_type, edge_index, h_mat):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, h_mat=h_mat)

    def message(self, edge_index_i, edge_index_j, node_inp_i, node_inp_j, node_type_i, node_type_j, h_mat):
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_att_h = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)

        for sub_type in range(self.num_types):
            sb = (node_type_i == int(sub_type))
            sub_linear = self.sub_linear[sub_type]

            for neigh_type in range(self.num_types):
                idx = (node_type_j == int(neigh_type)) & sb
                neigh_linear = self.neigh_linear[neigh_type]

                sub_node_vec = node_inp_i[idx]
                neigh_node_vec = node_inp_j[idx]

                h_sub = self.h_sub_att(h_mat[edge_index_i[idx]]).view(-1, self.n_heads, 8)
                h_neigh = self.h_neigh_att(h_mat[edge_index_j[idx]]).view(-1, self.n_heads, 8)
                res_att_h[idx] = (F.tanh((h_sub * h_neigh)) * self.relation_h_att[sub_type]).sum(-1) / self.d_h

                q_mat = sub_linear(sub_node_vec).view(-1, self.n_heads, self.d_k)
                k_mat = neigh_linear(neigh_node_vec).view(-1, self.n_heads, self.d_k)

                res_att[idx] = (F.tanh((q_mat * k_mat)) * self.relation_att[sub_type]).sum(-1) / self.sqrt_dk * \
                               res_att_h[idx]

                # v_mat = self.v_linear(neigh_node_vec).view(-1, self.n_heads, self.d_k)
                res_msg[idx] =k_mat

        self.att = softmax(res_att, edge_index_i)

        res = res_msg * self.att.view(-1, self.n_heads, 1)

        del res_att, res_msg
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_type):

        aggr_out = F.gelu(aggr_out)
        res = self.norm(aggr_out)
        return self.drop(res)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_node)


class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, num_types, n_heads, dropout):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'phgnn':
            self.base_conv = HGTConv(in_hid, out_hid, num_types, n_heads, dropout)
        elif self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)

    def forward(self, meta_xs, node_type, edge_index, h_mat):
        if self.conv_name == 'phgnn':
            return self.base_conv(meta_xs, node_type, edge_index, h_mat)
        elif self.conv_name == 'gcn':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'gat':
            return self.base_conv(meta_xs, edge_index)


