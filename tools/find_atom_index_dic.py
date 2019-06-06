#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:42:14 2019

@author: zl
"""

import os
import pandas as pd
import tqdm

def main():
    df_train = pd.read_csv(os.path.join('data', 'structures.csv'))
    atom_dic = {}
    for i, row in tqdm.tqdm(df_train.iterrows()):
        atom = row['atom']
        if atom not in atom_dic.keys():
            atom_dic[atom] = len(atom_dic.keys())
    
    atom_index_dir = 'data/atom_index_dic.txt'
    fileObject = open(atom_index_dir, 'w')
  #  for ip in key_group_list:
    fileObject.write(str(atom_dic))
    fileObject.write('\n')
    fileObject.close() 
        

if __name__ == '__main__':
    main()