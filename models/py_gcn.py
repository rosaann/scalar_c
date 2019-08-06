#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:34:47 2019

@author: zl
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv,NNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

n_features = 4
# definenet
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