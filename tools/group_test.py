#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 21:12:40 2019

@author: apple
"""

import os
import pandas as pd
import tqdm
import ast

def main():
    files = ['result_NetX_498.csv', 'result_NetX_752.csv', 'result_NetX_867.csv', 'result_NetX_959.csv']
    result = []
    out = []
    for fi, f in enumerate( files):       
        df = pd.read_csv(os.path.join('data', f))
        for i, row in tqdm.tqdm(df.iterrows()):
            c = row['scalar_coupling_constant']
            if i >= len(result):
                result.append(float(c))
            else:
                result[i] += c
    
            if fi == (len(files) - 1):
                out.append({'id' : int(row['id']), 'scalar_coupling_constant' : result[i] / len(files)})
    
    test_pd = pd.DataFrame.from_records(out, columns=['id', 'scalar_coupling_constant'])
    output_filename = os.path.join('data', 'result_NetX_s4.csv')       
    test_pd.to_csv(output_filename, index=False)    

if __name__ == '__main__':
   main()