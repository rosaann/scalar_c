#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:04:05 2019

@author: zl
"""

import os
import pandas as pd
import tqdm
import ast

def main():
    df_train = pd.read_csv(os.path.join('data', 'train.csv'))
    type_dic = {}
    for i, row in tqdm.tqdm(df_train.iterrows()):
        t = row['type']
        if t not in type_dic.keys():
            type_dic[t] = len(type_dic.keys())
    
    type_index_dir = 'data/type_index_dic.txt'
    fileObject = open(type_index_dir, 'w')
  #  for ip in key_group_list:
    fileObject.write(str(type_dic))
    fileObject.write('\n')
    fileObject.close() 
        

if __name__ == '__main__':
   main()
   