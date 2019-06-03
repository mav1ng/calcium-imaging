import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import h5py
import json
import matplotlib.pyplot as plt
from numpy import array, zeros
import numpy as np
from scipy.misc import imread
from glob import glob

import config as c
import network as n
import data
import corr

import json
import socket
import urllib.request as req

dtype = c.data['dtype']
device = c.cuda['device']

torch.cuda.empty_cache()

nf_corr = data.create_corr_data(neurofinder_path='data/neurofinder.00.00')
data.save_numpy_to_h5py(data_array=nf_corr['correlations'].numpy(), label_array=nf_corr['labels'].numpy(),
                        file_name='corr_nf_0000', file_path='data/corr/small_star/full/')

test_array = torch.rand(10, 32, 32)
corrs = corr.get_corr(test_array)
a_crop = test_array[:, :30, :30]
corrs_2 = corr.get_corr(a_crop)
corrs_3 = corrs[:, :30, :30]

comp = torch.where(corrs_3 == corrs_2, torch.tensor(0.), torch.tensor(1.))
indices = comp.nonzero()
print(indices)
