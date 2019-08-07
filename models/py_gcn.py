#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:34:47 2019

@author: zl
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv,NNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer


    
class EdgeModel_1(torch.nn.Module):
    def __init__(self):
        super(EdgeModel_1, self).__init__()
        self.edge_mlp = Sequential(Linear(9, 84), ReLU(), Linear(84, 39), ReLU())

    def forward(self, src, dest, edge_attr):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        print('src ', src.shape, ' ', src)
        print('dest ', dest.shape, ' ', dest)
        print('edge_attr ', edge_attr.shape, ' ', edge_attr)
        out = torch.cat([src, dest, edge_attr], 1)
        print('eage_model out 1', out.shape, ' ', out)
        out = self.edge_mlp(out)
        print('eage_model out 2', out.shape, ' ', out)
        return out

class NodeModel_1(torch.nn.Module):
    def __init__(self):
        super(NodeModel_1, self).__init__()
        self.node_mlp_1 = Sequential(Linear(43, 64), ReLU(), Linear(64, 128), ReLU())
        self.node_mlp_2 = Sequential(Linear(128,256), ReLU(), Linear(132, 256), ReLU())

    def forward(self, x, edge_index, edge_attr):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        print('x ', x.shape, ' ', x)
        print('edge_index ', edge_index.shape, ' ', edge_index)
        print('edge_attr ', edge_attr.shape, ' ', edge_attr)
        row, col = edge_index
        out = torch.cat([x[col], edge_attr], dim=1)
        print('node_model out 1', out.shape, ' ', out)
        out = self.node_mlp_1(out)
        print('node_model out 2', out.shape, ' ', out)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
        print('node_model out 3', out.shape, ' ', out)

        out = torch.cat([x, out], dim=1)
        print('node_model out 4', out.shape, ' ', out)
        out = self.node_mlp_2(out) 
        print('node_model out 5', out.shape, ' ', out)
        return out
class EdgeModel_2(torch.nn.Module):
    def __init__(self):
        super(EdgeModel_1, self).__init__()
        self.edge_mlp = Sequential(Linear(9, 84), ReLU(), Linear(34, 1), ReLU())

    def forward(self, src, dest, edge_attr):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        print('src2 ', src.shape, ' ', src)
        print('dest2 ', dest.shape, ' ', dest)
        print('edge_attr2 ', edge_attr.shape, ' ', edge_attr)
        out = torch.cat([src, dest, edge_attr], 1)
        print('eage_model out2 1', out.shape, ' ', out)
        out = self.edge_mlp(out)
        print('eage_model out2 2', out.shape, ' ', out)
        return out

class NodeModel_2(torch.nn.Module):
    def __init__(self):
        super(NodeModel_1, self).__init__()
        self.node_mlp_1 = Sequential(Linear(43, 20), ReLU(), Linear(64, 4), ReLU())
        
    def forward(self, x, edge_index, edge_attr):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        print('x2 ', x.shape, ' ', x)
        print('edge_index2 ', edge_index.shape, ' ', edge_index)
        print('edge_attr2 ', edge_attr.shape, ' ', edge_attr)
        row, col = edge_index
        out = torch.cat([x[col], edge_attr], dim=1)
        print('node_model out2 1', out.shape, ' ', out)
        out = self.node_mlp_1(out)
        print('node_model out2 2', out.shape, ' ', out)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
        print('node_model out2 3', out.shape, ' ', out)

        out = torch.cat([x, out], dim=1)
        print('node_model out2 4', out.shape, ' ', out)
        
        return out
class MetaLayer_t(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None):
        super(MetaLayer_t, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
   #     self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        """"""
        row, col = edge_index

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr)

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr)

        

        return x, edge_attr

    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
        self.node_model)
def PY_MetaLayer ():
    m = MetaLayer_t(EdgeModel(), NodeModel())
    return m



n_features = 4
# definenet
class PY_SYN_METAAS(torch.nn.Module):
    def __init__(self):
        super(PY_SYN_METAAS, self).__init__()
        self.meta1 = MetaLayer_t(EdgeModel_1(), NodeModel_1())
        
    def forward(self, data):
        x, edge_attr = self.meta1(data.x, data.edge_index, data.edge_attr)
        
class PY_GCG_NET(torch.nn.Module):
    def __init__(self):
        super(PY_GCG_NET, self).__init__()
        
        conv1 = GCNConv(1, 128, cached=False) # if you defined cache=True, the shape of batch must be same!
        self.nnconv1 = NNConv(n_features, 128, conv1)
        self.bn1 = BatchNorm1d(128)
        
        conv2 = GCNConv(128, 256, cached=False)
        self.nnconv2 = NNConv(128, 256, conv2)
        self.bn2 = BatchNorm1d(256)
        self.fc1 = Linear(256, 512)
      #  self.bn3 = BatchNorm1d(64)
        self.fc2 = Linear(512, 64)
        self.fc3 = Linear(64, 1)
         
    def forward(self, data):
        x, edge_index, edge_attr= data.x, data.edge_index, data.edge_attr
        print('x ', x)
        print('edge_index ', edge_index)
        print('edge_attr ', edge_attr)
        x = F.relu(self.nnconv1(x, edge_index, edge_attr))
        print('x1 ', x.shape)
        x = self.bn1(x)
        print('x2 ', x.shape)
        x = F.relu(self.nnconv2(x, edge_index))
        print('x3 ', x.shape)
        x = self.bn2(x)
        print('x4 ', x.shape)
        x = global_add_pool(x, data.batch)
        print('x5 ', x.shape)
        x = F.relu(self.fc1(x))
        print('x6 ', x.shape)
      #  x = self.bn3(x)
     #   print('x7 ', x.shape)
        x = F.relu(self.fc2(x))
        print('x8 ', x.shape)
        x = F.dropout(x, p=0.2, training=self.training)
        print('x9 ', x.shape)
        x = self.fc3(x)
        print('x10 ', x.shape)
        x = F.log_softmax(x, dim=1)
        print('x11 ', x.shape)
        return x       