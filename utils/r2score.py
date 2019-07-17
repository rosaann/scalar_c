#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:27:51 2019

@author: zl
"""

import torch
import math
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

class R2Score:
    def __init__(self,):
        print('dd')
        
    def count(self, tar_list, pre_list):
        print('target ', tar_list.shape, ' ',tar_list )
        tar_x = tar_list[:,np.newaxis]
      #  print('tar_x ', tar_x.shape)
        print('prediction ', pre_list.shape, ' ', pre_list)
        tar_x = tar_x.reshape(-1)
      #  print('tar_x_1 ', tar_x.shape)
        
        pre_list = pre_list.reshape(-1)
      #  print('pre_list ', pre_list.shape)
        total_r2 = r2_score(tar_x, pre_list)
        total_msq = mean_squared_error(tar_x, pre_list)
        total_mae = mean_absolute_error(tar_x, pre_list)
        
        x = tar_x.nonzero()
      #  print('x ', x)
        tar_used = tar_x[x]
      #  print('tar_used ', tar_used.shape, ' ', tar_used)
        prediction_used = pre_list[x]
      #  print('prediction_used ', prediction_used.shape, ' ', prediction_used)
        used_r2 = r2_score(tar_used, prediction_used)
        used_msq = mean_squared_error(tar_used, prediction_used)
        used_mae = mean_absolute_error(tar_used, prediction_used)
        
        
        return total_r2, used_r2, total_msq, used_msq, total_mae, used_mae