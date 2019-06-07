#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:15:17 2019

@author: apple
"""
from torch.utils.data.dataset import Dataset
import ast
import os
import tqdm
import pandas as pd

def get_default_info():
    info = []
    for i in range(29):
        #(x,y,z,atom_index)
        info.append(( -1, 100, 100, 100))
    return info
def main():
    atom_index_dic = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'F': 4}

    struc_csv_dir = 'data/structures.csv'
    train_csv_dir = 'data/train.csv'
    df_struc = pd.read_csv(struc_csv_dir)
    
    model_info_list = []
    pre_mol_name = ''
    model_info = get_default_info()
    for i, row in tqdm.tqdm(df_struc.iterrows()):
        molecule_name = row['molecule_name']
        if pre_mol_name != molecule_name:
            if len(model_info) > 0:
                model_info_list.append(model_info)
                print('model_info ', model_info)
            #开始一个新modedel
            model_info = get_default_info()
            pre_mol_name = molecule_name
            
        atom_index = int( row['atom_index'])
        atom = row['atom']
        x = row['x']
        y = row['y']
        z = row['z']
        print('ai ', atom_index, 'a ', atom, ' x ', x, ' y ', y, ' z ', z)
        model_info[atom_index] = (atom_index_dic[atom], x, y, z)
        if i > 80:
            break
   # print('re ', model_info_list)
if __name__ == '__main__':
  main()