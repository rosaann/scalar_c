#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:56:20 2019

@author: zl
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import torch.utils.data
import torch.utils.data.sampler
from torch.utils.data import DataLoader

#from .default import DefaultDataset, TestDataset
from .dgl_dataset import DGLDataset
#from .small import SmallDataset
#from .test import TestDataset
import dgl

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def get_gnu_dataloader(batch_size, split, **_):
    dataset = DGLDataset(split)

  #  dataloader = DataLoader(dataset,
  #                          shuffle=False,
  #                          batch_size=batch_size,
  #                          drop_last=False,
  #                          num_workers=6,
  #                          pin_memory=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=collate)

    return dataloader
