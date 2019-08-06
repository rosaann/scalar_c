from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import torch.utils.data
import torch.utils.data.sampler
#from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
#from .default import DefaultDataset, TestDataset
from .torch_geo_dataset import QL0
#from .dgl_dataset import DGLDataset
#from .small import SmallDataset
#from .test import TestDataset


def get_dataset(data):
    print('d_name ', data.name)
    f = globals().get(data.name)

    return f()


def get_dataloader( batch_size, split, **_):
    dataset = DefaultDataset(split)

    is_train = 'train' == split

    dataloader = DataLoader(dataset,
                            shuffle=is_train,
                            batch_size=batch_size,
                            drop_last=is_train,
                            num_workers=6,
                            pin_memory=False)
    return dataloader

def get_test_dataloader( batch_size, **_):
    dataset = TestDataset()

    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            drop_last=False,
                            num_workers=6,
                            pin_memory=False)
    return dataloader

def get_pygeo_dataloader(batch_size, split):
    dataset = QL0('ql0', split)
    dataloader = DataLoader(dataset,
                            
                            batch_size=batch_size,
                            shuffle=False,
                           )
    return dataloader
