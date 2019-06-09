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

max_atom = 29
def find_atom_index_dic():
    atom_index_dic_dir = 'data/atom_index_dic.txt'
    if os.path.exists(atom_index_dic_dir):
        with open(atom_index_dic_dir, 'r') as f: 
            atom_index_dic = ast.literal_eval(f.read())
            return atom_index_dic

atom_index_dic = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'F': 4}

def get_default_info():
    info = []
    for i in range(max_atom):
        #(x,y,z,atom_index)
        info.append(( -1, 100, 100, 100))
    return info

class DefaultDataset(Dataset):
    def __init__(self,
                 ):
        self.gen_stuc_set_list()
        self.gen_train_data()
        
    def gen_train_data(self):
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
            if pre_mol_name != molecule_name:
                if i > 0:
                    self.train_data_list.append(train_data)
                    self.gt_data_list.append(gt_data)
                train_data = self.model_info_set[molecule_name]
                gt_data = np.zeros((29, 29))
            index_0 = row['atom_index_0']
            index_1 = row['atom_index_1']
            scalar_coupling = row['scalar_coupling_constant']
            gt_data[int(index_0)][int(index_1)] = float(scalar_coupling)   
            if i > 30 :
                break
        self.train_data_list.append(train_data)
        self.gt_data_list.append(gt_data)
        
    def gen_stuc_set_list(self,):
        struc_csv_dir = 'data/structures.csv'
        
        df_struc = pd.read_csv(struc_csv_dir)
    
        self.model_info_set = {}
        model_info = []
        pre_mol_name = ''       
        molecule_name = ''
        for i, row in tqdm.tqdm(df_struc.iterrows()):
            molecule_name = row['molecule_name']
            if pre_mol_name != molecule_name:
                if len(model_info) > 0:
                    self.model_info_set[molecule_name] = model_info
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
            
        self.model_info_set[molecule_name] = model_info
        
def main():
    data_set = DefaultDataset()
    print('train_data_list ', data_set.train_data_list)
    print('gt_data_list ', data_set.gt_data_list)

    
if __name__ == '__main__':
  main()