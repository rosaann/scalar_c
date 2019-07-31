#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:52:23 2019

@author: zl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # 公式 (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 公式 (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        
    def build_weight(self, num_bases,num_rels, in_feat, out_feat):
        self.weight = nn.Parameter(torch.Tensor(num_bases, in_feat,
                                                out_feat))
        self.activation = F.relu
      #  print('self.weight_in ', self.weight_in.shape)
        if num_bases < num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp_in = nn.Parameter(torch.Tensor(num_rels, num_bases))
          #  print('self.w_comp_in ', self.w_comp_in.shape)
       
        self.bias = nn.Parameter(torch.Tensor(out_feat))
   #     print('self.bias ', self.bias_in.shape)
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight_in,
                                gain=nn.init.calculate_gain('relu'))
        
        nn.init.xavier_uniform_(self.w_comp_in,
                                    gain=nn.init.calculate_gain('relu'))
        
        nn.init.xavier_uniform_(self.bias_in,
                                    gain=nn.init.calculate_gain('relu'))
 
    def edge_attention(self, edges):
        # 公式 (2) 所需，边上的用户定义函数
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        a = self.attn_fc(z2)
        return {'e' : F.leaky_relu(a)}
 
    def message_func(self, edges):
        # 公式 (1)
        weight = self.weight_in.view(self.h_dim, self.num_bases, self.in_dim)
       #  print('---f--weight--1-in- ', weight.shape)
        weight = torch.matmul(self.w_comp_in, weight).view(self.num_rels,
                                                       self.h_dim, self.in_dim)
        #
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
        # 公式 (3), (4)所需，传递消息用的用户定义函数
        return {'z' : edges.src['z'], 'e' : edges.data['e']}
 
    def reduce_func(self, nodes):
        # 公式 (3), (4)所需, 归约用的用户定义函数
        # 公式 (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 公式 (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h' : h}
 
    def forward(self, h):
        
        z = self.fc(h)
        self.g.ndata['z'] = z
        # 公式 (2)
        self.g.apply_edges(self.edge_attention)
        # 公式 (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
    
    
class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge
 
    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs))
        
class GAT_X1(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT_X1, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)
 
    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h