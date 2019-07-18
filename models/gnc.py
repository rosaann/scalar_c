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
from functools import partial
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

class Reg_Old(nn.Module):
    def __init__(self):
        super(Regression_X1, self).__init__()
        in_dim = 4
        hidden_dim = 32
        
        self.gcn1 = GCN(in_dim, hidden_dim, F.relu)
        self.gcn2 = GCN(hidden_dim, hidden_dim, F.relu)
        self.gcn3 = GCN(hidden_dim, 1, F.relu)
      #  self.regression = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        print('g_reg ', g)
        x = self.gcn1(g)
        x = self.gcn2(x)
        x = self.gcn3(x)
    #    x = self.regression(x)
        print('x ', x)
        y = x.edata
        print('out ', y)
        return y
    
class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        
        print('self.weight ', self.weight.shape)
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
            print('self.w_comp ', self.w_comp.shape)
        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            print('self.bias ', self.bias.shape)
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            print('---f--weight--1-- ', weight.shape)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
            print('---f--weight--2-- ', weight.shape)
        else:
            weight = self.weight
            print('---f--weight--3-- ', weight.shape)

        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                print('embed ', embed.shape)
                index = edges.data['w'] * self.in_feat #+ edges.src['id']
                print('index ', index)
                msg = embed[index]
                print('msg --', msg.shape)
                return {'msg': msg}
        else:
            def message_func(edges):
                w = weight[edges.data['w']]
                print('w ', w.shape)
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                print('msg ', msg.shape)
               # msg = msg * edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            h2 = nodes.data['h2']
            print('h2 ', h2.shape)
            h = nodes.data['h']
            print('h ', h.shape)
            
            if self.bias:
                h = h + self.bias
                print('h1 ', h.shape)
            if self.activation:
                h = self.activation(h)
                print('h2 ', h.shape)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h2'), apply_func)
        
class Regression_X1(nn.Module):
    def __init__(self, in_dim = 4, h_dim = 48, out_dim = 1, num_rels = 1,
                 num_bases=-1, num_hidden_layers=2):
        super(Regression_X1, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # create rgcn layers
        self.build_model()

        # create initial features
      #  self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.in_dim)
        return features

    def build_input_layer(self):
        return RGCNLayer(self.in_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases
                         )

    def forward(self, g):
       # if self.features is not None:
       #     g.ndata['id'] = self.features
        for i, layer in enumerate( self.layers):
            print('f-- ', i)
            layer(g)
        return g.ndata.pop('h')
