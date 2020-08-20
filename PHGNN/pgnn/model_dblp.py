from conv_dblp import *
import torch.nn as nn
import torch
import torch.nn.functional as F
class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.linear = nn.Linear(n_hid, n_out)

    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)

    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)




class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, n_heads, n_layers, dropout=0.2, conv_name='hgt'):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        self.norm=nn.LayerNorm(n_hid)
        self.linear=nn.ModuleList()
        self.skip = nn.Parameter(torch.ones(n_layers))
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim[t], n_hid ))
        for l in range(n_layers):
            self.gcs.append(GeneralConv(conv_name,n_hid, n_hid, num_types, n_heads, dropout))
            self.linear.append(nn.Linear((l+1)* n_hid, n_hid))
    def forward(self, features_0,features_1,features_2,features_3, node_type, edge_index, h_mat):
        preview=[]
        res = torch.zeros(node_type.size(0), self.n_hid ).to(node_type.device)

        features=[features_0,features_1,features_2,features_3,]
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](features[t_id]))
        meta_xs = self.drop(res)
        preview.append(meta_xs)
        del res
        for l,gc in enumerate(self.gcs):
            # meta_xs=torch.cat(preview,1)
            # meta_xs=self.norm(F.gelu(self.linear[l](meta_xs)))
            meta_xs = gc(meta_xs, node_type, edge_index, h_mat)
            # preview.append(meta_xs*self.skip[l])
        return meta_xs
