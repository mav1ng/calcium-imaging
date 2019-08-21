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

# """AZRAEL OPT ROUND 2.0"""
# """Taking Parameters from Optimisation Round 1"""
# nb_epochs = 250
# lr = 0.001
# bs = 3
# emb_dim = 128
# sub_sample_size = 1024
# for margin in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
#     print('Embedding Dim: ', emb_dim, 'Margin:', margin, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#     set = h.Setup(model_name='azrael2_' + str(emb_dim) + '_' + str(margin),
#                   embedding_dim=emb_dim, margin=margin, nb_epochs=nb_epochs, save_config=True, learning_rate=lr,
#                   batch_size=bs, include_background=True,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()
# ana.score('azrael2_', include_metric=True)

# """EVE OPT ROUND 4"""
# nb_epochs = 250
# lr = 0.00045
# bs = 4
# subsample_size = 1024
# for i in range(1, 6):
#     cur_emb_dim = np.random.randint(65, 128)
#     cur_margin = np.random.randint(1, 10) / 10
#     cur_scaling = 25
#     print('Embedding Dim: ', cur_emb_dim, 'Margin: ', cur_margin, 'Scaling: ', cur_scaling, 'Number epochs: ',
#           nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#     set = h.Setup(model_name='eve2_' + str(cur_emb_dim) + '_' + str(cur_margin) + '_' + str(cur_scaling),
#                   embedding_dim=cur_emb_dim, margin=cur_margin, scaling=cur_scaling,
#                   nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=True,
#                   background_pred=True, subsample_size=subsample_size,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()
# ana.score('eve2_', include_metric=True)


# set = h.Setup(model_name='m_eve_opt',
#               subsample_size=4, embedding_dim=63, margin=0.3, nb_epochs=27,
#               save_config=True, learning_rate=0.01, scaling=800,
#               batch_size=1, include_background=True,
#               background_pred=True,
#               nb_iterations=0, embedding_loss=True)
# set.main()
# ana.score('m_eve_opt', include_metric=True)

# h.val_score(model_name='m_eve_opt', use_metric=True, iter=10, th=0.8)
# h.val_score(model_name='m_eve3_4', use_metric=True, iter=10, th=0.8)
# h.test(model_name='m_eve3_4')
# h.test(model_name='ezekiel_opt')

# h.val_score('m_abram_opt', use_metric=True)
h.test('ezekiel_0.01497_14_9')



# ana.save_images('cain')

# ana.plot_analysis('eve2_114_0.3_25', th=0)

ana.analysis(analysis='lr_ep_bs', analysis_name='ezekiel_0', use_metric=False)
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



