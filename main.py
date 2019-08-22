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

# data.synchronise_folder()


"""EVE OPT ROUND X"""
emb_dim = 256
margin = 0.5

scaling_list = np.linspace(10, 30, 300)
lr_list = np.linspace(0.0001, 0.006, 10000)
subsample_size_list = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]) ** 2

for i in range(10):
    nb_epochs = np.random.randint(150, 251)
    scaling = np.around(np.random.choice(scaling_list), decimals=2)
    bs = np.random.randint(1, 5)
    lr = np.around(np.random.choice(lr_list), decimals=5)
    subsample_size = np.random.choice(subsample_size_list)

    print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin: ', margin, 'Scaling: ', scaling,
          'Number epochs: ',
          nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs)
    set = h.Setup(
        model_name='evex_' + str(subsample_size) + '_' + str(lr) + '_' + str(bs) + '_' + str(scaling) + '_' + str(
            nb_epochs),
        subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, scaling=scaling,
        nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=True,
        background_pred=True,
        nb_iterations=0, embedding_loss=True)
    set.main()
# ana.score('evex_', include_metric=True)
# ana.save_images('evex_')


"""EVE OPT ROUND X"""
emb_dim = 256
margin = 0.5

scaling_list = np.linspace(10, 30, 300)
lr_list = np.linspace(0.0001, 0.1, 10000)
subsample_size_list = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]) ** 2

for i in range(100):
    nb_epochs = np.random.randint(10, 251)
    scaling = np.around(np.random.choice(scaling_list), decimals=2)
    bs = np.random.randint(1, 11)
    lr = np.around(np.random.choice(lr_list), decimals=5)
    subsample_size = np.random.choice(subsample_size_list)

    print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin: ', margin, 'Scaling: ', scaling,
          'Number epochs: ',
          nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs)
    set = h.Setup(
        model_name='evex_' + str(subsample_size) + '_' + str(lr) + '_' + str(bs) + '_' + str(scaling) + '_' + str(
            nb_epochs),
        subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, scaling=scaling,
        nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=True,
        background_pred=True,
        nb_iterations=0, embedding_loss=True)
    set.main()
ana.score('evex_', include_metric=True)
ana.save_images('evex_')



# set = h.Setup(model_name='m_eve_opt',
#               subsample_size=4, embedding_dim=63, margin=0.3, nb_epochs=27,
#               save_config=True, learning_rate=0.01, scaling=800,
#               batch_size=1, include_background=True,
#               background_pred=True,
#               nb_iterations=0, embedding_loss=True)
# set.main()
# ana.score('m_eve_opt', include_metric=True)







# data.synchronise_folder()

# ana.score('eve2_', include_metric=True)

# h.val_score(model_name='m_eve_opt', use_metric=True, iter=10, th=0.8)
# h.val_score(model_name='m_eve3_4', use_metric=True, iter=10, th=0.8)
# h.test(model_name='m_eve3_4')
# h.test(model_name='ezekiel_opt')

# h.val_score('m_abram_opt', use_metric=True)
# h.test('ezekiel_0.01497_14_9')



# ana.save_images('eve')

# ana.plot_analysis('eve2_114_0.3_25', th=0)

# ana.analysis(analysis='lr_ep_bs', analysis_name='m_abel', use_metric=False)
# ana.analysis(analysis='ed_ma_sc', analysis_name='m_adam2_', use_metric=True)
# ana.analysis(analysis='ed_ma', analysis_name='m_azrael2_', use_metric=True)
# ana.analysis(analysis='ss', analysis_name='m_eve3_', use_metric=True)
# ana.analysis(analysis='lr', analysis_name='m_adam4_', use_metric=True)

# ana.score('eve', include_metric=True)
# ana.score('cain', include_metric=True)
# ana.score('adam3')
#
# ana.score('azrael3_')
# ana.score('eve3_')
# ana.score('ezekiel3_')



