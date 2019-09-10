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
# h.test('abram_0.0068_11')
# h.test('mean_shift_full_test')

"""NOAH OPT ROUND 1"""
margin = 0.5
nb_epochs = 100
nb_iter = 5
step_size = 1.

kernel_bandwidth_list = np.linspace(5, 15, 1000)
scaling_list = np.linspace(1, 10, 300)
lr_list = np.linspace(0.0001, 0.01, 10000)
subsample_size = 1024

for i in range(50):
    kernel_bandwidth = np.around(np.random.choice(kernel_bandwidth_list), decimals=2)
    emb_dim = np.random.randint(8, 33)
    scaling = np.around(np.random.choice(scaling_list), decimals=2)
    bs = np.random.randint(1, 21)
    lr = np.around(np.random.choice(lr_list), decimals=5)

    print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin: ', margin, 'Scaling: ', scaling,
          'Number epochs: ',
          nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs, 'kernel_bandwidth', kernel_bandwidth)
    set = h.Setup(
        model_name='noah_' + str(lr) + '_' + str(bs) + '_' + str(scaling) + '_' + str(kernel_bandwidth),
        subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, scaling=scaling,
        nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=False,
        background_pred=True,
        nb_iterations=nb_iter, kernel_bandwidth=kernel_bandwidth, step_size=step_size, embedding_loss=True)
    set.main()
ana.score('noah_', include_metric=True)
ana.save_images('noah_')


# set = h.Setup(
#         model_name='mean_shift_full_test',
#         subsample_size=1024, embedding_dim=16, margin=0.5, scaling=5,
#         nb_epochs=100, save_config=True, learning_rate=0.0005, batch_size=4, include_background=False,
#         background_pred=True,
#         nb_iterations=5, kernel_bandwidth=10., step_size=1., embedding_loss=True)
# set.main()
# ana.score('mean_shift_full_test', include_metric=True)
# ana.save_images('mean_shift_full_test')

# set = h.Setup(model_name='ms_test_1',
#               subsample_size=1024, embedding_dim=3, margin=0.5, nb_epochs=10,
#               save_config=True, learning_rate=0.002, scaling=10,
#               batch_size=1, include_background=False, kernel_bandwidth=12., step_size=1.,
#               background_pred=True,
#               nb_iterations=5, embedding_loss=True)
# set.main()



# set = h.Setup(model_name='emb_test',
#               subsample_size=1024, embedding_dim=16, margin=0.5, nb_epochs=1000,
#               save_config=True, learning_rate=0.002, scaling=10,
#               batch_size=5, include_background=False, kernel_bandwidth=None, step_size=1.,
#               background_pred=True,
#               nb_iterations=0, embedding_loss=True)
# set.main()


# set = h.Setup(model_name='emb_test',
#               subsample_size=1024, embedding_dim=16, margin=0.5, nb_epochs=10,
#               save_config=True, learning_rate=0.002, scaling=10,
#               batch_size=5, include_background=False, kernel_bandwidth=None, step_size=1.,
#               background_pred=True,
#               nb_iterations=0, embedding_loss=True)
# set.main()
# h.test('t2')

#
# set = h.Setup(model_name='t2e',
#               subsample_size=1024, embedding_dim=3, margin=0.5, nb_epochs=10,
#               save_config=True, learning_rate=0.0001, scaling=0,
#               batch_size=1, include_background=False, kernel_bandwidth=None, step_size=1.,
#               background_pred=True,
#               nb_iterations=5, embedding_loss=True, pre_train=True, pre_train_name='t_pre')
# set.main()
# h.test('t2e')

# set = h.Setup(model_name='t_pre',
#               subsample_size=1024, embedding_dim=3, margin=0.5, nb_epochs=1,
#               save_config=True, learning_rate=0., scaling=0,
#               batch_size=1, include_background=True, kernel_bandwidth=None, step_size=1.,
#               background_pred=True,
#               nb_iterations=0, embedding_loss=False)
# set.main()


# set = h.Setup(model_name='abram_test_short',
#               subsample_size=1024, embedding_dim=3, margin=0.5, nb_epochs=50,
#               save_config=True, learning_rate=0.0005, scaling=10,
#               batch_size=1, include_background=False, kernel_bandwidth=None, step_size=1.,
#               background_pred=True,
#               nb_iterations=0, embedding_loss=False)
# set.main()


# set = h.Setup(model_name='test_4.3',
#               subsample_size=1024, embedding_dim=18, margin=0.5, nb_epochs=12,
#               save_config=True, learning_rate=0.0005, scaling=0,
#               batch_size=1, include_background=False, kernel_bandwidth=None, step_size=1.,
#               background_pred=True,
#               nb_iterations=0, embedding_loss=True)
# set.main()


# set = h.Setup(model_name='test',
#               subsample_size=100, embedding_dim=32, margin=0.5, nb_epochs=2,
#               save_config=True, learning_rate=0.001, scaling=25,
#               batch_size=10, include_background=True, kernel_bandwidth=3., step_size=1.,
#               background_pred=True,
#               nb_iterations=10, embedding_loss=True)
# set.main()

# data.get_summary_img(nf_folder='data/test_data', sum_folder='data/test_sum_img',
#                        test=True, device=torch.device('cpu'), dtype=torch.double)


# h.det_bandwidth(model_name='evey_1024_0.00072_4_8.16_208_22')



# ana.score('adamx_', include_metric=True)
# ana.save_images('adamx_')

# data.synchronise_folder()

# ana.score('eve2_', include_metric=True)

# h.val_score(model_name='m_eve_opt', use_metric=True, iter=10, th=0.8)
# h.val_score(model_name='m_eve3_4', use_metric=True, iter=10, th=0.8)
# h.test(model_name='m_eve3_4')
# h.test(model_name='ezekiel_opt')

# h.val_score('adamy_784_0.00117_4_13.63_157_17', use_metric=True, th=0.8)
# h.test('evey_900_0.00035_4_5.5_249_18')

# h.val_score('evey_676_0.00065_4_14.28_192_18', use_metric=True, th=0.8)
# h.test('evey_1024_0.00072_4_8.16_208_22')



# ana.save_images('noahy', postproc=False)

# ana.plot_analysis('eve2_114_0.3_25', th=0)



# ana.analysis(analysis='lr_ep_bs', analysis_name='eve', use_metric=False)
# ana.analysis(analysis='ed_ma_sc', analysis_name='eve', use_metric=False)
# ana.analysis(analysis='ed_ma', analysis_name='abram', use_metric=False)
# ana.analysis(analysis='ss', analysis_name='evey', use_metric=True)
# ana.analysis(analysis='lr', analysis_name='m_adam4_', use_metric=True)

# ana.score('noah', include_metric=False, iter=3)

# ana.score('cain', include_metric=True)
# ana.score('adam3')
#
# ana.score('azrael3_')
# ana.score('eve3_')
# ana.score('ezekiel3_')

