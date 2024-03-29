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
import os, ast
from functools import partial
import numpy as np
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

def find_type_index_dic():
    type_index_dic_dir = 'data/type_index_dic.txt'
    if os.path.exists(type_index_dic_dir):
        with open(type_index_dic_dir, 'r') as f: 
            type_index_dic_dir = ast.literal_eval(f.read())
            return type_index_dic_dir 
        
class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self,num_ref, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
       # self.linear = nn.Linear(in_feats, out_feats)
        self.weight_whole = nn.Parameter(torch.Tensor(num_ref, out_feats,
                                                in_feats)).cuda()
        self.activation = activation

    def forward(self, node):
       # h = self.linear(node.data['h'])
        print('node ', node)
        print('weight_whole ', self.weight_whole.shape)
        node_d = node.data['h']
        print('node_d ', node_d.shape)
        edge_d = node.edata['we']
        print('edge_d ', edge_d.shape, ' ', edge_d)
        
        h =  torch.bmm( node_d, edge_d)
        print('h ', h.shape)
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.type_index_dic = find_type_index_dic()
        self.num_rels = len( self.type_index_dic.keys())
        self.apply_mod = NodeApplyModule(self.num_rels, in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
               
        g.update_all(msg, reduce)
        
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')
     #   return g.ndata.pop('h')

class Regression_X1_Up(nn.Module):
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
                                                self.out_feat)).cuda()
        
        print('self.weight ', self.weight.shape)
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases).cuda())
            print('self.w_comp ', self.w_comp.shape)
        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat).cuda())
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
                node = edges.src['h']
                
                print('node ', node)
                print('node  s ', node.shape)
                embed = weight.view(-1, self.out_feat)
                print('embed ', embed.shape)
                print('edge w ', edges.data['w'].shape)
                print('self.in_feat ', self.in_feat)
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
    def __init__(self, in_dim = 4, h_dim = 256, out_dim = 1, num_rels = 1,
                 num_bases=1, num_hidden_layers=2):
        super(Regression_X1, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        #self.num_rels = num_rels
        self.type_index_dic = find_type_index_dic()
        self.num_rels = len( self.type_index_dic.keys())
      #  print('self.num_rels ', self.num_rels)
        self.num_hidden_layers = num_hidden_layers
        self.num_bases = num_bases
        # create rgcn layers
        self.build_input_layer(in_dim, h_dim, self.num_rels, num_bases)
        self.build_h0_layer(h_dim, h_dim, self.num_rels, num_bases)
        self.build_out_layer(h_dim, out_dim, self.num_rels, num_bases)
      


    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.in_dim)
        return features

    def build_input_layer(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        self.weight_in = nn.Parameter(torch.Tensor(num_bases, in_feat,
                                                out_feat))
        self.activation_in = F.relu
      #  print('self.weight_in ', self.weight_in.shape)
        if num_bases < num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp_in = nn.Parameter(torch.Tensor(num_rels, num_bases))
          #  print('self.w_comp_in ', self.w_comp_in.shape)
       
      #  self.bias_in = nn.Parameter(torch.Tensor(out_feat))
   #     print('self.bias ', self.bias_in.shape)
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight_in,
                                gain=nn.init.calculate_gain('relu'))
        if num_bases < num_rels:
            nn.init.xavier_uniform_(self.w_comp_in,
                                    gain=nn.init.calculate_gain('relu'))
        if bias:
            nn.init.xavier_uniform_(self.bias_in,
                                    gain=nn.init.calculate_gain('relu'))
            
    
    def forward_in(self, g):
         
            # generate all weights from bases (equation (3))
         weight = self.weight_in.view(self.h_dim, self.num_bases, self.in_dim)
       #  print('---f--weight--1-in- ', weight.shape)
         weight = torch.matmul(self.w_comp_in, weight).view(self.num_rels,
                                                       self.h_dim, self.in_dim)
      #   print('---f--weight--2-in- ', weight.shape) 
     #    bias_in = self.bias_in
         activation_in = self.activation_in
         def edge_message_func_in(edges):
                print('edge_message_func_in ', edges) 
                msg = ''
                index = edges.data['we'] 
                wd = edges.data['wd']
                
                for i,real_idx in enumerate(index):                   
                   # print('real_idx', real_idx)
                    embed = weight[real_idx]
                  #  print('embed ', embed.shape)
 
                    w = embed[0]
                 #   print('w ', w.shape, ' ')
                    node_data = wd[i]
                    node_data = torch.unsqueeze (node_data, 0)
                    
                 #   print('node_data ', node_data.shape, ' ')
                    if i == 0:
                        msg = F.linear(node_data, w)
                    else:
                        msg = torch.cat((msg, F.linear(node_data, w)), 0)
                    
              #  print('msg ', msg.shape, ' ')
                return {'r':msg}
         def message_func_in(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
               
              #  print('message_func_in ', edges) 
              #  print('edges d ', edges.data['we'])
             #   print('edges src ', edges.src['h'])
              #  print('edges des ', edges.dst['h'])
                msg = ''
                index = edges.data['we'] 
                src = edges.src['h']
             #   wd = edges.data['wd']
                
                for i,real_idx in enumerate(index):                   
                   # print('real_idx', real_idx)
                    embed = weight[real_idx]
                  #  print('embed ', embed.shape)
 
                    w = embed[0]
                 #   print('w ', w.shape, ' ')
                    node_data = src[i]
                    node_data = torch.unsqueeze (node_data, 0)
                    
                 #   print('node_data ', node_data.shape, ' ')
                    if i == 0:
                        msg = F.linear(node_data, w)
                    else:
                        msg = torch.cat((msg, F.linear(node_data, w)), 0)
             #   edges.data['r'] = msg
              #  print('msg ', msg.shape, ' ')

                return {'msg':msg}
        
         def reduce_func_in(nodes):
          #   print('reduce_func_in nodes ',  nodes)
         #    print('nodes.mailbox', nodes.mailbox['msg'])
             msg = torch.sum(nodes.mailbox['msg'], dim=1)
             
             return {'h2' : msg}
         
         def apply_func_in(nodes):
          #  h2 = nodes.data['h2']
         #   print('apply_func_in ', nodes)
            h = nodes.data['h2']
         #   print('h ', h.shape)
           
          #  h = h + bias_in
         #   print('h1 ', h.shape)
            h = activation_in(h)
         #   print('h2 ', h.shape)
            return {'h2': h}
         g.update_all(message_func_in, reduce_func_in, apply_func_in)
        # g.apply_edges(edge_message_func_in)
       #  result = g.edata.pop('r')
        # print('result ', result)
         return g
        # g.update_all(message_func_in, reduce_func_in, apply_func_in)
    
    def build_h0_layer(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        self.weight_h0 = nn.Parameter(torch.Tensor(num_bases, in_feat,
                                                out_feat))
        self.activation_h0 = F.relu
      #  print('self.weight_in ', self.weight_in.shape)
        if num_bases < num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp_h0 = nn.Parameter(torch.Tensor(num_rels, num_bases))
          #  print('self.w_comp_in ', self.w_comp_in.shape)
       
    #    self.bias_h0 = nn.Parameter(torch.Tensor(out_feat))
   #     print('self.bias ', self.bias_in.shape)
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight_h0,
                                gain=nn.init.calculate_gain('relu'))
        if num_bases < num_rels:
            nn.init.xavier_uniform_(self.w_comp_h0,
                                    gain=nn.init.calculate_gain('relu'))
        if bias:
            nn.init.xavier_uniform_(self.bias_h0,
                                    gain=nn.init.calculate_gain('relu'))
            
    
    def forward_h0(self, g):
         
            # generate all weights from bases (equation (3))
         weight = self.weight_h0.view(self.h_dim, self.num_bases, self.h_dim)
       #  print('---f--weight--1-in- ', weight.shape)
         weight = torch.matmul(self.w_comp_h0, weight).view(self.num_rels,
                                                       self.h_dim, self.h_dim)
      #   print('---f--weight--2-in- ', weight.shape) 
     #    bias_h0 = self.bias_h0
         activation_h0 = self.activation_h0
         def message_func_h0(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
               
              #  print('message_func_in ', edges) 
              #  print('edges d ', edges.data['we'])
             #   print('edges src ', edges.src['h'])
              #  print('edges des ', edges.dst['h'])
                msg = ''
                index = edges.data['we'] 
                src = edges.src['h2']
             #   wd = edges.data['wd']
                
                for i,real_idx in enumerate(index):                   
                   # print('real_idx', real_idx)
                    embed = weight[real_idx]
                  #  print('embed ', embed.shape)
 
                    w = embed[0]
                 #   print('w ', w.shape, ' ')
                    node_data = src[i]
                    node_data = torch.unsqueeze (node_data, 0)
                    
                 #   print('node_data ', node_data.shape, ' ')
                    if i == 0:
                        msg = F.linear(node_data, w)
                    else:
                        msg = torch.cat((msg, F.linear(node_data, w)), 0)
               # edges.data['r'] = msg
              #  print('msg ', msg.shape, ' ')

                return {'msg':msg}
        
         def reduce_func_h0(nodes):
           #  print('reduce_func_in nodes ',  nodes)
          #   print('nodes.mailbox', nodes.mailbox['msg'])
             msg = torch.sum(nodes.mailbox['msg'], dim=1)
             
             return {'h2' : msg}
         
         def apply_func_h0(nodes):
          #  h2 = nodes.data['h2']
          #  print('apply_func_in ', nodes)
            h = nodes.data['h2']
         #   print('h ', h.shape)
           
         #   h = h + bias_h0
         #   print('h1 ', h.shape)
            h = activation_h0(h)
          #  print('h2 ', h.shape)
            return {'h2': h}
         g.update_all(message_func_h0, reduce_func_h0, apply_func_h0)
        # g.apply_edges(edge_message_func_in)
       #  result = g.edata.pop('r')
        # print('result ', result)
         return g

      #   g.update_all(message_func_h0, fn.sum(msg='msg', out='h2'), apply_func_h0)

    def build_out_layer(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        self.weight_out = nn.Parameter(torch.Tensor(num_bases, in_feat,
                                                out_feat))
        self.activation_out = F.relu
      #  print('self.weight_in ', self.weight_in.shape)
        if num_bases < num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp_out = nn.Parameter(torch.Tensor(num_rels, num_bases))
          #  print('self.w_comp_in ', self.w_comp_in.shape)
       
      #  self.bias_out = nn.Parameter(torch.Tensor(out_feat))
   #     print('self.bias ', self.bias_in.shape)
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight_out,
                                gain=nn.init.calculate_gain('relu'))
        if num_bases < num_rels:
            nn.init.xavier_uniform_(self.w_comp_out,
                                    gain=nn.init.calculate_gain('relu'))
        if bias:
            nn.init.xavier_uniform_(self.bias_out,
                                    gain=nn.init.calculate_gain('relu'))
            
    
    def forward_out(self, g):
         
            # generate all weights from bases (equation (3))
         weight = self.weight_out.view(self.out_dim, self.num_bases, self.h_dim)
       #  print('---f--weight--1-in- ', weight.shape)
         weight = torch.matmul(self.w_comp_out, weight).view(self.num_rels,
                                                       self.out_dim, self.h_dim)
      #   print('---f--weight--2-in- ', weight.shape) 
     #    bias_out = self.bias_out
         activation_out = self.activation_out
         def message_func_out(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
               
              #  print('message_func_in ', edges) 
              #  print('edges d ', edges.data['we'])
             #   print('edges src ', edges.src['h'])
              #  print('edges des ', edges.dst['h'])
                msg = ''
                index = edges.data['we'] 
                src = edges.src['h2']
             #   wd = edges.data['wd']
                
                for i,real_idx in enumerate(index):                   
                   # print('real_idx', real_idx)
                    embed = weight[real_idx]
                  #  print('embed ', embed.shape)
 
                    w = embed[0]
                 #   print('w ', w.shape, ' ')
                    node_data = src[i]
                    node_data = torch.unsqueeze (node_data, 0)
                    
                 #   print('node_data ', node_data.shape, ' ')
                    if i == 0:
                        msg = F.linear(node_data, w)
                    else:
                        msg = torch.cat((msg, F.linear(node_data, w)), 0)
                edges.data['r'] = msg
              #  print('msg ', msg.shape, ' ')

                return {'msg':msg}
        
         def reduce_func_out(nodes):
         #    print('reduce_func_in nodes ',  nodes)
         #    print('nodes.mailbox', nodes.mailbox['msg'])
             msg = torch.sum(nodes.mailbox['msg'], dim=1)
             
             return {'h2' : msg}
         
         def apply_func_out(nodes):
          #  h2 = nodes.data['h2']
         #   print('apply_func_in ', nodes)
            h = nodes.data['h2']
         #   print('h ', h.shape)
           
        #    h = h + bias_out
        #    print('h1 ', h.shape)
            h = activation_out(h)
        #    print('h2 ', h.shape)
            return 
         g.update_all(message_func_out, reduce_func_out, apply_func_out)
        # g.apply_edges(edge_message_func_in)
         g.ndata.pop('h2')
         result = g.edata.pop('r')
        # print('result ', result)
         return result
    

    def forward(self, g):
       # if self.features is not None:
       #     g.ndata['id'] = self.features
        g = self.forward_in(g)
        g = self.forward_h0(g)
        r = self.forward_out(g)
       
     #   print('r ', r)
        return r   