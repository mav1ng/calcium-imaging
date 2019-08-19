import sys

for i, path in enumerate(sys.path):
    sys.path.pop(i)

for path in ['/net/hcihome/storage/mvspreng/PycharmProjects/calcium-imaging',
             '/export/home/mvspreng/anaconda3/envs/pytorch/lib/python37.zip',
             '/export/home/mvspreng/anaconda3/envs/pytorch/lib/python3.7',
             '/export/home/mvspreng/anaconda3/envs/pytorch/lib/python3.7/lib-dynload',
             '/export/home/mvspreng/anaconda3/envs/pytorch/lib/python3.7/site-packages']:
    if path not in sys.path:
        sys.path.append(path)

from torch import optim
import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter
import umap
import torch.nn as nn

import config as c
import analysis as ana
import data
import corr
import network as n
import visualization as v
import training as t
import clustering as cl
import helpers as h
import argparse
import time
import random
import numpy as np
import neurofinder as nf

from torchsummary import summary

"""METRIC ABEL OPT ROUND 1"""
for i in range(25, 51):
    cur_nb_epochs = np.random.randint(10, 251)
    cur_lr = np.random.randint(1, 10001) / 100000.
    cur_bs = int(np.ceil(i / 10))
    print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
    set = h.Setup(model_name='m_abel_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs), embedding_dim=32,
                  nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs,
                  include_background=True,
                  background_pred=True, scaling=25, margin=0.5,
                  nb_iterations=3, kernel_bandwidth=None, step_size=1., embedding_loss=True)
    set.main()
ana.score('m_abel_', include_metric=True)

# h.val_score(model_name='eve2_22_0.2_947.2', use_metric=True, iter=100, th=0.8)
# h.test(model_name='m_azrael_opt')
# h.test(model_name='eve2_22_0.2_947.2')

# ana.analysis(analysis='lr_ep_bs', analysis_name='m_azrael_', use_metric=True)
# ana.analysis(analysis='ed_ma_sc', analysis_name='m_adam2_', use_metric=True)
# ana.analysis(analysis='ed_ma', analysis_name='m_azrael2_', use_metric=True)
# ana.analysis(analysis='ss', analysis_name='m_azrael3_', use_metric=True)

# ana.score('abram_t_')
# ana.score('adam3')
#
# ana.score('azrael3_')
# ana.score('eve3_')
# ana.score('ezekiel3_')

