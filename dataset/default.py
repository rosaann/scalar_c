#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:34:22 2019

@author: zl
"""

from torch.utils.data.dataset import Dataset
import ast
import os
import tqdm
import pandas as pd
import numpy as np
import random
import cmath

max_atom = 30
def find_atom_index_dic():
    atom_index_dic_dir = 'data/atom_index_dic.txt'
    if os.path.exists(atom_index_dic_dir):
        with open(atom_index_dic_dir, 'r') as f: 
            atom_index_dic = ast.literal_eval(f.read())
            return atom_index_dic

atom_index_dic = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'F': 4}

def get_default_info():
    info = []
    for i in range(max_atom ):
        #(x,y,z,atom_index)
        info.append(( -1, 100, 100, 100))
   # info = np.array(info)
   # print('info xx ', info.shape)
   # info = info.reshape(max_atom, max_atom, 4)
    return info
def save_data_to_local(file, data):
    fileObject = open(file, 'w')
  #  for ip in key_group_list:
    fileObject.write(str(data))
    fileObject.write('\n')
    fileObject.close() 
    
def changeListFormToRectForm(data_list):
    size = len(data_list)
    result = []
    rlist = []
    for in range(size):
        rlist.append(size * size)
        
    for i in range(size * size):
        if i in rlist:
            num_sqrt = i
            if i != 0:
                num_sqrt = cmath.sqrt(i)
            result.append(data_list[num_sqrt])
        else:
            result.append(( -1, 100, 100, 100))
            
    result = np.array(result)
    
    return result.reshape(size, size)
        
class DefaultDataset(Dataset):
    def __init__(self,split
                 ):
        print('**DefaultDataset ')
        self.gen_stuc_set_list()
        
        self.gen_train_data()
        print('self.gen_train_data()')
        self.gen_random_index_list()
        print('self.gen_random_index_list ')
        self.split = split
        self.data_list = []
        n7 = int (len(self.random_index_list) * 0.7)
        if split == 'train':
            self.gt_list = []
            for i in self.random_index_list[ : n7]:
                d_p = changeListFormToRectForm(self.train_data_list[self.random_index_list[i]])
                self.data_list.append(d_p)
                self.gt_list.append(self.gt_data_list[self.random_index_list[i]])
            self.data_list = np.array(self.data_list)
            tshape = self.data_list.shape
            print('self.data_list ', tshape)
            self.data_list = self.data_list.reshape(tshape[0],tshape[2], tshape[1])
            print('self.data_list2 ', self.data_list.shape)
            self.gt_list = np.array(self.gt_list)
            print('self.gt_list ', self.gt_list.shape)
        
        if split == 'val':
            for i in self.random_index_list[n7 : ]:
                self.data_list.append(self.train_data_list[self.random_index_list[i]])
            self.data_list = np.array(self.data_list)
    def __getitem__(self, index):
        if self.split == 'train':
            return {'data': self.data_list[index],
                'gt': self.gt_list[index]
                }  
        else :
             return {'data': self.data_list[index]} 
    
    def __len__(self):
        return len( self.data_list)    
    
    def gen_train_data(self):
        text_file_train_all = 'data/train_data_list.txt'
        text_file_gt_all = 'data/gt_data_list.txt'
        
        if os.path.exists(text_file_train_all):
            with open(text_file_train_all, 'r') as f: 
                self.train_data_list = ast.literal_eval(f.read())
        
        if os.path.exists(text_file_gt_all):
            with open(text_file_gt_all, 'r') as f: 
                self.gt_data_list = ast.literal_eval(f.read())
                return
        
        train_csv_dir = 'data/train.csv'
        df_train = pd.read_csv(train_csv_dir)
        self.train_data_list = []
        self.gt_data_list = []
        
        pre_mol_name = ''       
        molecule_name = ''
        train_data = ''
        gt_data = ''
        for i, row in tqdm.tqdm(df_train.iterrows()):
            molecule_name = row['molecule_name']
         #   print('molecule_name ', molecule_name)
            if pre_mol_name != molecule_name:
                if i > 0:
                    self.train_data_list.append(train_data)
                    self.gt_data_list.append(gt_data.tolist())
                 #   print('gt_data save ', gt_data)
                train_data = self.model_info_set[molecule_name]
                gt_data = np.zeros((max_atom, max_atom))
                pre_mol_name = molecule_name
            index_0 = row['atom_index_0']
            index_1 = row['atom_index_1']
            scalar_coupling = row['scalar_coupling_constant']
         #   print('scalar_coupling ', scalar_coupling)
         #   print('int(index_0) ', int(index_0))
         #   print('int(index_1) ', int(index_1))
            gt_data[int(index_0)][int(index_1)] = float(scalar_coupling)   
         #   print('gt_data ', gt_data)
          #  if i > 30 :
          #      print('self.train_data_list ', self.train_data_list)
          #      break
        self.train_data_list.append(train_data)
        save_data_to_local(text_file_train_all, self.train_data_list)
        
        self.gt_data_list.append(gt_data.tolist())
        save_data_to_local(text_file_gt_all, self.gt_data_list)
    
    def gen_random_index_list(self):
        txt_random_index_file = 'data/random_index_list.txt'
        if os.path.exists(txt_random_index_file):
            with open(txt_random_index_file, 'r') as f: 
                self.random_index_list = ast.literal_eval(f.read())
                return
        num = len(self.train_data_list)
        self.random_index_list = list(range(num))
        random.shuffle(self.random_index_list)
        save_data_to_local(txt_random_index_file, self.random_index_list)
        
    def gen_stuc_set_list(self,):
            
        txt_file = 'data/model_info_set.txt'
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f: 
                self.model_info_set = ast.literal_eval(f.read())
                return
        
        struc_csv_dir = 'data/structures.csv'
        df_struc = pd.read_csv(struc_csv_dir)
    
        self.model_info_set = {}
        model_info = []
        pre_mol_name = ''       
        molecule_name = ''
        for i, row in tqdm.tqdm(df_struc.iterrows()):
            molecule_name = row['molecule_name']
          #  print('molecule_name ', molecule_name)
            if pre_mol_name != molecule_name:
                if i > 0:
                 #   print('molecule_name s ', pre_mol_name)
                    self.model_info_set[pre_mol_name] = model_info
                #    print('model_info ', model_info)
                #开始一个新modedel
                model_info = get_default_info()
                pre_mol_name = molecule_name
            
            atom_index = int( row['atom_index'])
            atom = row['atom']
            x = row['x']
            y = row['y']
            z = row['z']
         #   print('ai ', atom_index, 'a ', atom, ' x ', x, ' y ', y, ' z ', z)
            model_info[atom_index] = (atom_index_dic[atom], x, y, z)
        #    if i > 20:
         #       break
        self.model_info_set[molecule_name] = model_info
        save_data_to_local(txt_file, self.model_info_set)
        
def main():
    data_set = DefaultDataset()
  #  print('train_data_list ', data_set.train_data_list)
   # print('gt_data_list ', data_set.gt_data_list)

    
if __name__ == '__main__':
  main()