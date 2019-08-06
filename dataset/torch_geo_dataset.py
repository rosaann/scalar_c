#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:02:30 2019

@author: zl
"""

import torch
from torch_geometric.data import Data
import os, ast
import tqdm
import pandas as pd
import numpy as np
import random
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
                                  Data)
from itertools import repeat, product
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
         
            if i > 10:
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

class QL0(InMemoryDataset):
    def __init__(self,
                 root,
                 split = 'train',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.split = split
        super(QL0, self).__init__(root, transform, pre_transform, pre_filter)
        
        print('split ', self.split)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return 'ql0.pt'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        return

    def process(self):
        n7 = int (len(random_index_list) * 0.7)
        index_list = random_index_list[ : n7]
        if self.split == 'val':
            index_list = random_index_list[n7 : ]
        data_list = []
        for i in index_list:
             d_data = train_data_list[random_index_list[i]]
             nodes = d_data['nodes']
             edges = d_data['edges']
             
             d = []      
             for node_info in nodes:
                 #   idx = int(node_info['idx'])
                    tp = int(node_info['t'])
                    x = float(node_info['x'])
                    y = float(node_info['y'])
                    z = float(node_info['z'])
                    dn = [tp, x, y, z]
                 #   n = torch.tensor( dn).cuda()
                    d.append(dn)
                    
                    
             e_attr = []
             
             e_idx= []
             gt = []
             for edge_info in edges:
                    idx0 = int(edge_info['index0'])
                    idx1 = int(edge_info['index1'])
                    et = int(edge_info['et'])
                    sc = float(edge_info['sc'])
                    e_idx.append([idx0, idx1])
                    e_at= [et, d[idx0][0], d[idx0][1], d[idx0][2], d[idx0][3]]
                    print('e_at ', e_at)
                    e_attr.append(e_at)
                    gt.append(sc)
                    
          #   geo_data = Data(x = d, edge_index =e_idx, edge_attr = e_attr, y=gt )
             edge_index = torch.tensor(e_idx)
             x = torch.tensor(d, dtype=torch.float)
             print('eattr ', e_attr)
             e_attr = torch.tensor(e_attr, dtype=torch.float)
             gt = torch.tensor(gt, dtype=torch.float)
           #  geo_data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr = e_attr, y=gt)
             geo_data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr = e_attr, y=gt)
             print('geo_data-- ', geo_data)
             data_list.append(geo_data)
        

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate_t(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def collate_t(self, data_list):
        r"""Collates a python list of data objects to the internal storage
        format of :class:`torch_geometric.data.InMemoryDataset`."""
        keys = data_list[0].keys
        data = data_list[0].__class__()

        for key in keys:
            data[key] = []
        slices = {key: [0] for key in keys}

        for item, key in product(data_list, keys):
            print('key ', key)
            print('item ', item)
            data[key].append(item[key])
            if torch.is_tensor(item[key]):
                s = slices[key][-1] + item[key].size(
                    item.__cat_dim__(key, item[key]))
            elif isinstance(item[key], int) or isinstance(item[key], float):
                s = slices[key][-1] + 1
            else:
                raise ValueError('Unsupported attribute type')
            slices[key].append(s)

        if hasattr(data_list[0], '__num_nodes__'):
            data.__num_nodes__ = []
            for item in data_list:
                data.__num_nodes__.append(item.num_nodes)

        for key in keys:
            if torch.is_tensor(data_list[0][key]):
                data[key] = torch.cat(
                    data[key], dim=data.__cat_dim__(key, data_list[0][key]))
            else:
                data[key] = torch.tensor(data[key])
            slices[key] = torch.tensor(slices[key], dtype=torch.long)

        return data, slices


