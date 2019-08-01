#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:52:23 2019

@author: zl
"""
import torch, os, ast
import torch.nn as nn
import torch.nn.functional as F

def find_type_index_dic():
    type_index_dic_dir = 'data/type_index_dic.txt'
    if os.path.exists(type_index_dic_dir):
        with open(type_index_dic_dir, 'r') as f: 
            type_index_dic_dir = ast.literal_eval(f.read())
            return type_index_dic_dir 
        
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, num_bases = 1):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        
       # self.g = g
        # 公式 (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 公式 (2)
        self.attn_fc = nn.Linear(2 * in_dim, 1, bias=False)
        self.build_weight(in_dim, out_dim, num_rels, num_bases)
        
        
    def build_weight(self, in_feat, out_feat, num_rels, num_bases):
        self.w = nn.Parameter(torch.Tensor(num_bases, in_feat,
                                                out_feat))
        self.activation = F.relu
      #  print('self.weight_in ', self.weight_in.shape)
       # if num_bases < num_rels:
            # linear combination coefficients in equation (3)
        self.w_comp = nn.Parameter(torch.Tensor(num_rels, num_bases))
          #  print('self.w_comp_in ', self.w_comp_in.shape)
       
    #    self.bias = nn.Parameter(torch.Tensor(out_feat))
   #     print('self.bias ', self.bias_in.shape)
        # init trainable parameters
        nn.init.xavier_uniform_(self.w,
                                gain=nn.init.calculate_gain('relu'))
        
        nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        
    #    nn.init.xavier_uniform_(self.bias_in,
    #                                gain=nn.init.calculate_gain('relu'))
 
        
    def edge_attention(self, edges):
        # 公式 (2) 所需，边上的用户定义函数
      #  print('es ', edges.src['h'].shape)
      #  print('ed ', edges.dst['h'].shape)
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
    #    print('z2 ', z2.shape, ' ', z2)
        a = self.attn_fc(z2)
        return {'e' : F.leaky_relu(a)}
 
    def message_func(self, edges):
        weight = self.w.view(self.out_dim, self.num_bases, self.in_dim)
       #  print('---f--weight--1-in- ', weight.shape)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                       self.out_dim, self.in_dim)
        
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
        edges.data['r'] = msg
        return {'msg':msg, 'e' : edges.data['e']}
        # 公式 (3), (4)所需，传递消息用的用户定义函数
        
      #  return {'z' : edges.src['z'], 'e' : edges.data['e']}
 
    def reduce_func(self, nodes):
        # 公式 (3), (4)所需, 归约用的用户定义函数
        # 公式 (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
     #   print('alpha ', alpha.shape, ' ', alpha)
        # 公式 (4)
        h = torch.sum(alpha * nodes.mailbox['msg'], dim=1)
     #   print('h ', h.shape, ' ', h)
        return {'h' : h}
 
    def forward(self, g):
        h = g.ndata['h']
        z = self.fc(h)
        g.ndata['z'] = z
        # 公式 (2)
        g.apply_edges(self.edge_attention)
        # 公式 (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')
    
    
class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_ref, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer( in_dim, out_dim, num_ref))
        self.merge = merge
 
    def forward(self, g):
       
        head_outs = [attn_head(g) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            g.ndata['h'] = torch.cat(head_outs, dim=1)
            return g
        else:
            # 用求平均整合多头结果
            g.ndata['h'] = torch.mean(torch.stack(head_outs))
            return g
        
class GAT_X1(nn.Module):
    def __init__(self, in_dim = 4, hidden_dim = 128, out_dim = 1, num_heads = 1):
        super(GAT_X1, self).__init__()
        self.type_index_dic = find_type_index_dic()
        self.num_rels = len( self.type_index_dic.keys())
        
        self.layer1 = MultiHeadGATLayer( in_dim, hidden_dim, num_heads, self.num_rels)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        self.layer2 = MultiHeadGATLayer( hidden_dim * num_heads, out_dim, 1, self.num_rels)
 
    def forward(self, g):
    #    print('forward 1')
        g = self.layer1(g)
   #     print('forward 2')
        g.ndata['h'] = F.elu(g.ndata['h'])
        
        g = self.layer2(g)
   #     print('forward 3')
        g.ndata.pop('h')
        result = g.edata.pop('r')
        return result