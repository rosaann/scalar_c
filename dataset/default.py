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
import time
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
    for i in range(size):
        rlist.append(i * size + i)
        
    for i in range(size * size):
        if i in rlist:
            num_sqrt = i % size           
            result.append(data_list[num_sqrt])
        else:
            result.append(( -1, 100, 100, 100))
            
    result = np.array(result)
    
    return result.reshape(size, size, 4)
        
 

def get_gt_data():
        text_file_gt_all = 'data/gt_data_list.txt'
        if os.path.exists(text_file_gt_all):
            f = open(text_file_gt_all, 'r') 
            gt_data_list = ast.literal_eval(f.read())
            f.close() 
            return gt_data_list
def gen_test_data():
        text_file_test_all = 'data/test_data_list.txt'   
        if os.path.exists(text_file_test_all):
            f = open(text_file_test_all, 'r') 
            test_data_list = ast.literal_eval(f.read())
            f.close() 
        
            return test_data_list   
        test_csv_dir = 'data/test.csv'
        df_test = pd.read_csv(test_csv_dir)
        test_data_list = []
        
        pre_mol_name = ''       
        molecule_name = ''
        test_data = ''
        for i, row in tqdm.tqdm(df_test.iterrows()):
            molecule_name = row['molecule_name']
         #   print('molecule_name ', molecule_name)
            if pre_mol_name != molecule_name:
                if i > 0:
                    test_data_list.append({'name':pre_mol_name, 'data' : test_data})
                    
                test_data = model_info_set[molecule_name]
                pre_mol_name = molecule_name
            
         #   print('gt_data ', gt_data)
          #  if i > 30 :
          #      print('self.train_data_list ', self.train_data_list)
          #      break
        test_data_list.append({'name':pre_mol_name, 'data' : test_data})
        save_data_to_local(text_file_test_all, test_data_list)
        return test_data_list
        
def gen_train_data():
        text_file_train_all = 'data/train_data_list.txt'
        text_file_gt_all = 'data/gt_data_list.txt'
        
        if os.path.exists(text_file_train_all):
            f = open(text_file_train_all, 'r') 
            train_data_list = ast.literal_eval(f.read())
            f.close() 
        
            return train_data_list
        
        train_csv_dir = 'data/train.csv'
        df_train = pd.read_csv(train_csv_dir)
        train_data_list = []
        gt_data_list = []
        
        pre_mol_name = ''       
        molecule_name = ''
        train_data = ''
        gt_data = ''
        for i, row in tqdm.tqdm(df_train.iterrows()):
            molecule_name = row['molecule_name']
         #   print('molecule_name ', molecule_name)
            if pre_mol_name != molecule_name:
                if i > 0:
                    train_data_list.append(train_data)
                    gt_data_list.append(gt_data.tolist())
                 #   print('gt_data save ', gt_data)
                train_data = model_info_set[molecule_name]
                gt_data = np.zeros((max_atom, max_atom ))
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
        train_data_list.append(train_data)
        save_data_to_local(text_file_train_all, train_data_list)
        
        gt_data_list.append(gt_data.tolist())
        save_data_to_local(text_file_gt_all, gt_data_list)
    
def gen_random_index_list():
        txt_random_index_file = 'data/random_index_list.txt'
        if os.path.exists(txt_random_index_file):
            f = open(txt_random_index_file, 'r') 
            random_index_list = ast.literal_eval(f.read())
            f.close() 
            return random_index_list
        num = len(train_data_list)
        random_index_list = list(range(num))
        random.shuffle(random_index_list)
        save_data_to_local(txt_random_index_file, random_index_list)

def gen_stuc_set_list():
            
        txt_file = 'data/model_info_set.txt'
        if os.path.exists(txt_file):
            f = open(txt_file, 'r')  
            model_info_set = ast.literal_eval(f.read())
            f.close() 
            return model_info_set
        
        struc_csv_dir = 'data/structures.csv'
        df_struc = pd.read_csv(struc_csv_dir)
    
        model_info_set = {}
        model_info = []
        pre_mol_name = ''       
        molecule_name = ''
        for i, row in tqdm.tqdm(df_struc.iterrows()):
            molecule_name = row['molecule_name']
          #  print('molecule_name ', molecule_name)
            if pre_mol_name != molecule_name:
                if i > 0:
                 #   print('molecule_name s ', pre_mol_name)
                    model_info_set[pre_mol_name] = model_info
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
        model_info_set[molecule_name] = model_info
        save_data_to_local(txt_file, model_info_set)
        

print('**DefaultDataset ')
model_info_set = gen_stuc_set_list()

print('self.gen_stuc_set_list()')
gt_data_list = get_gt_data()
       
print('self.get_gt_data()')
train_data_list = gen_train_data()
       
print('self.gen_train_data()')
        
random_index_list = gen_random_index_list()
        
print('self.gen_random_index_list ')    
test_data_list = gen_test_data()

class TestDataset(Dataset):
    def __init__(self
                 ):
         self.data_list = []
         for i in range( len(test_data_list)):
                data_con = test_data_list[i]
                print('data_con ', data_con)
                d_p = changeListFormToRectForm(data_con['data'])
                
                self.data_list.append({'data': d_p, 'name': test_data_list[i]['name']})
         self.data_list = np.array(self.data_list)
         tshape = self.data_list.shape
         self.data_list = self.data_list.reshape(tshape[0],tshape[3], tshape[1], tshape[2])
         
    def __getitem__(self, index):
       
            return  self.data_list[index]
                
                  
    def __len__(self):
        return len( self.data_list)  

class DefaultDataset(Dataset):
    def __init__(self,split
                 ):
        
        self.split = split
        self.data_list = []
        self.gt_list = []
        n7 = int (len(random_index_list) * 0.7)
        print('enter DefaultDataset ', split)
        if split == 'train':
            
            for i in random_index_list[ : n7]:
                d_p = changeListFormToRectForm(train_data_list[random_index_list[i]])
              #  if i == 0:
               #     print('d_p ', d_p)
                self.data_list.append(d_p)
                self.gt_list.append(gt_data_list[random_index_list[i]])
            self.tr_data_list = np.array(self.data_list)
            tshape = self.tr_data_list.shape
            print('self.data_list ', tshape)
            self.tr_data_list = self.tr_data_list.reshape(tshape[0],tshape[3], tshape[1], tshape[2])
            print('self.data_list2 ', self.tr_data_list.shape)
            self.gt_list = np.array(self.gt_list)
            print('self.gt_list ', self.gt_list.shape)
        
        if split == 'val':
            self.val_data_list = []
            for i in random_index_list[n7 : ]:
                self.val_data_list.append(changeListFormToRectForm(train_data_list[random_index_list[i]]))
                self.gt_list.append(gt_data_list[random_index_list[i]])
            self.val_data_list = np.array(self.val_data_list)
            tshape = self.val_data_list.shape
            print('self.data_list ', tshape)
            self.val_data_list = self.val_data_list.reshape(tshape[0],tshape[3], tshape[1], tshape[2])
            print('self.data_list2 ', self.val_data_list.shape)
            self.val_data_list = np.array(self.val_data_list)
            self.gt_list = np.array(self.gt_list)
            print('self.gt_list ', self.gt_list.shape)
    def setSplit(self, split):
        self.split = split
    def __getitem__(self, index):
       # if index % 10 == 0:
         #   print('index ', index)
        if self.split == 'train':
            return {'data': self.tr_data_list[index],
                'gt': self.gt_list[index]
                }  
        else :
             return {'data': self.val_data_list[index],
                     'gt': self.gt_list[index]} 
    
    def __len__(self):
        return len( self.gt_list)      
def main():
    data_set = DefaultDataset()
  #  print('train_data_list ', data_set.train_data_list)
   # print('gt_data_list ', data_set.gt_data_list)

    
if __name__ == '__main__':
  main()