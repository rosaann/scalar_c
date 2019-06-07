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

def find_atom_index_dic():
    atom_index_dic_dir = 'data/atom_index_dic.txt'
    if os.path.exists(atom_index_dic_dir):
        with open(atom_index_dic_dir, 'r') as f: 
            atom_index_dic = ast.literal_eval(f.read())
            return atom_index_dic

atom_index_dic = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'F': 4}

def get_default_info():
    info = []
    for i in range(29):
        #(x,y,z,atom_index)
        info.append(( -1, 100, 100, 100))
    return info

class DefaultDataset(Dataset):
    def __init__(self,
                 ):
        struc_csv_dir = 'data/scalar_c/data/structures.csv'
        train_csv_dir = 'data/scalar_c/data/train.csv'
        df_struc = pd.read_csv(struc_csv_dir)
    
        self.model_info_list = []
        pre_mol_name = ''
        model_info = []
        for i, row in tqdm.tqdm(df_struc.iterrows()):
            molecule_name = row['molecule_name']
            if molecule_name != pre_mol_name:
                if len(model_info) > 0:
                    self.model_info_list.append(model_info)
                #开始一个新modedel
                model_info = get_default_info()
            
            atom_index = int( row['atom_index'])
            atom = row['atom']
            x = row['x']
            y = row['y']
            z = row['z']
            model_info[atom_index] = (atom_index_dic[atom], x, y, z)