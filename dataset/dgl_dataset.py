#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:39:45 2019

@author: zl
"""

from torch.utils.data.dataset import Dataset
from dgl.graph import DGLGraph
import ast
import os
import tqdm
import pandas as pd
import numpy as np
import random
import cmath
import time
import torch

max_atom = 29


def save_data_to_local(file, data):
    fileObject = open(file, 'w')
  #  for ip in key_group_list:
    fileObject.write(str(data))
    fileObject.write('\n')
    fileObject.close() 
    
def find_atom_index_dic():
    atom_index_dic_dir = 'data/atom_index_dic.txt'
    if os.path.exists(atom_index_dic_dir):
        with open(atom_index_dic_dir, 'r') as f: 
            atom_index_dic = ast.literal_eval(f.read())
            return atom_index_dic
def find_type_index_dic():
    type_index_dic_dir = 'data/type_index_dic.txt'
    if os.path.exists(type_index_dic_dir):
        with open(type_index_dic_dir, 'r') as f: 
            type_index_dic_dir = ast.literal_eval(f.read())
            return type_index_dic_dir
        
        
atom_index_dic = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'F': 4}
type_index_dic = find_type_index_dic()

def gen_stuc_set_list():
        
        
        struc_csv_dir = 'data/structures.csv'
        df_struc = pd.read_csv(struc_csv_dir)
        
    
        model_info_set = {}
        model_info_list = []
        pre_mol_name = ''       
        molecule_name = ''
        for i, row in tqdm.tqdm(df_struc.iterrows()):
            molecule_name = row['molecule_name']
          #  print('molecule_name ', molecule_name)
            if pre_mol_name != molecule_name:
                if i > 0:
                 #   print('molecule_name s ', pre_mol_name)
                    model_info_set[pre_mol_name] ={'nodes': model_info_list}
                #    print('model_info ', model_info)
                #开始一个新modedel
                model_info_list = []
                pre_mol_name = molecule_name
            
            atom_index = int( row['atom_index'])
            atom = row['atom']
            x = row['x']
            y = row['y']
            z = row['z']
         #   print('ai ', atom_index, 'a ', atom, ' x ', x, ' y ', y, ' z ', z)
            info = {'idx': atom_index, 't': atom_index_dic[atom], 'x': x, 'y' : y, 'z' : z}
            model_info_list.append(info)
         #   if i > 200:
         #       break
        model_info_set[molecule_name] = model_info_list
        
       # np2 = int(len())
      #  for i, path in enumerate(txt_file):
       #     save_data_to_local(path, model_info_set)
        return model_info_set

def gen_train_data():
     
        train_csv_dir = 'data/train.csv'
        df_train = pd.read_csv(train_csv_dir)
        train_data_list = []
        
        pre_mol_name = ''       
        molecule_name = ''
        train_data = ''
        
       # print('model_info_set ', model_info_set)
        for i, row in tqdm.tqdm(df_train.iterrows()):
            molecule_name = row['molecule_name']
            t = row['type']
            t_index = type_index_dic[t]
         #   print('molecule_name ', molecule_name)
            if pre_mol_name != molecule_name:
                if i > 0:
                    train_data_list.append(train_data)
               
                train_data = model_info_set[molecule_name]
                train_data['edges'] = []
               
                pre_mol_name = molecule_name
            index_0 = row['atom_index_0']
            index_1 = row['atom_index_1']
            
            scalar_coupling = row['scalar_coupling_constant']            
            train_data['edges'].append({'index0' : index_0, 'index1' : index_1, 'et' : t_index, 'sc' : float(scalar_coupling)})
         
            if i > 200:
                break
        train_data_list.append(train_data)
        
        return train_data_list

def gen_random_index_list():
        txt_random_index_file = 'data/dgl_random_index_list.txt'
        if os.path.exists(txt_random_index_file):
            f = open(txt_random_index_file, 'r') 
            random_index_list = ast.literal_eval(f.read())
            f.close() 
            return random_index_list
        num = len(train_data_list)
        random_index_list = list(range(num))
        random.shuffle(random_index_list)
        save_data_to_local(txt_random_index_file, random_index_list)
        return random_index_list


print('**DGL_DefaultDataset ')
model_info_set = gen_stuc_set_list()
print('self.gen_stuc_set_list()')

train_data_list = gen_train_data()
print('self.gen_train_data()')     
  
random_index_list = gen_random_index_list()
        
print('self.gen_random_index_list ')  

    
class DGLDataset(object):
    def __init__(self, split
                 ):
        super(DGLDataset, self).__init__()
        self.device = torch.device("cuda" )
        self.split = split
        self.data_list = []
        self.gt_list = []
        n7 = int (len(random_index_list) * 0.7)
        print('enter DGLDataset ', random_index_list)
        if split == 'train':
            
            for i in random_index_list[ : n7]:
                d_data = train_data_list[random_index_list[i]]
                if i == 0:
                    print('d_data ', d_data)
                nodes = d_data['nodes']
                edges = d_data['edges']
                
                g = DGLGraph()
                g.add_nodes(len(nodes))
                gt = []
            #    {'idx': atom_index, 't': atom_index_dic[atom], 'x': x, 'y' : y, 'z' : z}
                for node_info in nodes:
                    idx = int(node_info['idx'])
                    tp = int(node_info['t'])
                    x = float(node_info['x'])
                    y = float(node_info['y'])
                    z = float(node_info['z'])
                    g.nodes[idx].data['h'] = torch.tensor( [[tp, x, y, z]]).cuda()
                    
                
             #   gt = []
                e = []
                for edge_info in edges:
                    idx0 = int(edge_info['index0'])
                    idx1 = int(edge_info['index1'])
                    et = int(edge_info['et'])
                    sc = float(edge_info['sc'])
                    g.add_edge(idx0, idx1)
                    e.append([idx0, idx1, et])
                 #   if 'w' not in g.edata.keys():
                 #       g.edata['w'] =  torch.tensor( [[et]]).cuda()
                #    else :
                #        g.edata['w'].expand( torch.tensor( [et]).cuda())
                    gt.append(sc)
                print('e ', e)
                g.edata['we'] = torch.tensor(e).cuda()
                print('g ', g)   
                
                self.data_list.append(g)
                self.gt_list.append(gt)
              #  self.gt_list.append([gt])
               # self.gt_list.append(1)
            print('len data ', len(self.data_list))  
            print('len gt ', len(self.gt_list))
        #    self.gt_list = np.array(self.gt_list)
         #   self.data_list = np.array(self.data_list)
         #   tshape = self.data_list.shape
        #    print('self.data_list ', tshape)
            
        if split == 'val':
            self.val_data_list = []
            for i in random_index_list[n7 : ]:
                d_data = train_data_list[random_index_list[i]]
                if i == 0:
                    print('d_data ', d_data)
                nodes = d_data['nodes']
                edges = d_data['edges']
                
                g = DGLGraph()
                g.add_nodes(len(nodes))
                gt = []
            #    {'idx': atom_index, 't': atom_index_dic[atom], 'x': x, 'y' : y, 'z' : z}
                for node_info in nodes:
                    idx = int(node_info['idx'])
                    tp = int(node_info['t'])
                    x = float(node_info['x'])
                    y = float(node_info['y'])
                    z = float(node_info['z'])
                    g.nodes[idx].data['h'] = torch.tensor( [[tp, x, y, z]]).cuda()
                    
                
             #   gt = []
                for edge_info in edges:
                    idx0 = int(edge_info['index0'])
                    idx1 = int(edge_info['index1'])
                    et = int(edge_info['et'])
                    sc = float(edge_info['sc'])
                    g.add_edge(idx0, idx1)
                    g.edges[idx0, idx1].data['w'] = torch.tensor( [et]).cuda()
                    gt.append(sc)
                    
                
                self.data_list.append(g)
                self.gt_list.append(gt)
                
            print('len v data ', len(self.data_list))  
            print('len v gt ', len(self.gt_list))
        #    self.gt_list = np.array(self.gt_list)
         #   self.data_list = np.array(self.data_list)
         #   tshape = self.data_list.shape
         #   print('self.data_list ', tshape)
        #self.g_multi.g_multi.add_nodes(len( df_struc.iterrows() ))

        
        
    def __getitem__(self, index):
      #  if self.split == 'train':
            return self.data_list[index], self.gt_list[index]
      #  else:
      #      return self.data_list[index]
        
    def __len__(self):
        return len( self.data_list)  