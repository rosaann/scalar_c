from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import torch.utils.data
import torch.utils.data.sampler
from torch.utils.data import DataLoader

from .default import DefaultDataset
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
