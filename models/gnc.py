#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:20:06 2019

@author: zl
"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
import networkx as nx

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)
def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g):
        # Initialize the node features with h.
       # g.ndata['h'] = feature
   #     print('ggg ', g)
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g
     #   return g.ndata.pop('h')

class Regression_X1(nn.Module):
    def __init__(self):
        super(Regression_X1, self).__init__()
        in_dim = 4
        hidden_dim = 32
        
        self.gcn1 = GCN(in_dim, hidden_dim, F.relu)
        self.gcn2 = GCN(hidden_dim, hidden_dim, F.relu)
      #  self.regression = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        print('g_reg ', g)
        x = self.gcn1(g)
        x = self.gcn2(g)
    #    x = self.regression(x)
        y = x.data['h']
        print('out ', y)
        return y
    

