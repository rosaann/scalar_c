#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:58:20 2019

@author: zl
"""

import os
import pandas as pd
import tqdm

def main():
    df_train = pd.read_csv(os.path.join('data', 'structures.csv'))
    max_atom_index = 0
    for i, row in tqdm.tqdm(df_train.iterrows()):
        atom_index = row['atom_index']
        if atom_index > max_atom_index:
            max_atom_index = atom_index
            
    print('max atom index ', max_atom_index)

if __name__ == '__main__':
    main()