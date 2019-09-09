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
# h.test('128_test_long')


"""ABRAM OPT ROUND 1"""
cur_nb_epochs = 100
lr_list = np.linspace(0.0001, 0.01, 10000)

for i in range(100):
    cur_emb = np.random.choice(np.array([0]))
    cur_bs = np.random.randint(1, 21)
    cur_lr = np.around(np.random.choice(lr_list), decimals=5)
    print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs, 'Emb Dim: ', cur_emb)
    set = h.Setup(model_name='abram' + '_' + str(cur_lr) + '_' + str(cur_bs) + '_' + str(cur_emb),
                  nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs,
                  embedding_loss=False, background_pred=True, nb_iterations=0, embedding_dim=cur_emb)
    set.main()
ana.score('abram_', include_metric=True)
ana.save_images('abram_')


# set = h.Setup(model_name='128_test_short',
#               subsample_size=1024, embedding_dim=18, margin=0.5, nb_epochs=50,
#               save_config=True, learning_rate=0.01, scaling=10,
#               batch_size=4, include_background=False, kernel_bandwidth=None, step_size=1.,
#               background_pred=True,
#               nb_iterations=0, embedding_loss=False)
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

