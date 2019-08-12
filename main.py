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

# set = h.Setup(model_name='test2', nb_epochs=1000)
# set.main()

# ana.analysis(analysis='lr_ep_bs', analysis_name='eve_')

# ana.score('abram_')
# ana.score('adam_')

# ana.score('azrael_')
# ana.score('eve_')
ana.score('ezekiel_')


# """ABRAM OPT ROUND 1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10000) / 100000.
#     cur_bs = int(np.ceil(i / 10))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='abram_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, embedding_loss=False,
#                   background_pred=True)
#     set.main()

# """ABRAM OPT ROUND 1.1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(150, 251)
#     cur_lr = np.random.randint(1, 1000) / 100000.
#     cur_bs = int(np.ceil(i / 25))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='abram_1.1_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, embedding_loss=False,
#                   background_pred=True)
#     set.main()


# """ABRAM OPT ROUND 1.2"""
# for i in range(1, 50):
#     cur_nb_epochs = np.random.randint(200, 251)
#     cur_lr = np.random.randint(1, 1000) / 100000.
#     cur_bs = int(np.ceil(i / 25))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='abram_1.2_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, embedding_loss=False,
#                   background_pred=True)
#     set.main()


# """AZRAEL OPT ROUND 1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10000) / 100000.
#     cur_bs = int(np.ceil(i / 10))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='azrael_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=True,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

# """AZRAEL OPT ROUND 1.1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(150, 251)
#     cur_lr = np.random.randint(1, 1000) / 100000.
#     cur_bs = int(np.ceil(i / 25))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='azrael_1.1_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=True,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

# """EZEKIEL OPT ROUND 1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10000) / 100000.
#     cur_bs = int(np.ceil(i / 10))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='ezekiel_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=False,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

# """EZEKIEL OPT ROUND 1.1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(150, 251)
#     cur_lr = np.random.randint(1, 1000) / 100000.
#     cur_bs = int(np.ceil(i / 25))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='ezekiel_1.1_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=False,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()


# """EVE OPT ROUND 1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10000) / 100000.
#     cur_bs = int(np.ceil(i / 10))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='eve_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=True,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

# """EVE OPT ROUND 1.1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(150, 251)
#     cur_lr = np.random.randint(1, 1000) / 100000.
#     cur_bs = int(np.ceil(i / 25))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='eve_1.1_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=True,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()


# """ADAM OPT ROUND 1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10000) / 100000.
#     cur_bs = int(np.ceil(i / 10))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='adam_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=False,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()


# """ADAM OPT ROUND 1.1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(150, 251)
#     cur_lr = np.random.randint(1, 1000) / 100000.
#     cur_bs = int(np.ceil(i / 25))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='adam_1.1_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=False,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

