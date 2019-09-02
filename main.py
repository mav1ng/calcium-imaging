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


# set = h.Setup(model_name='test',
#               subsample_size=100, embedding_dim=32, margin=0.5, nb_epochs=2,
#               save_config=True, learning_rate=0.001, scaling=25,
#               batch_size=10, include_background=True, kernel_bandwidth=3., step_size=1.,
#               background_pred=True,
#               nb_iterations=10, embedding_loss=True)
# set.main()

# data.get_summary_img(nf_folder='data/test_data', sum_folder='data/test_sum_img',
#                        test=True, device=torch.device('cpu'), dtype=torch.double)


# h.det_bandwidth(model_name='adamx_676_0.00286_2_13.68_182')



# ana.score('adamx_', include_metric=True)
# ana.save_images('adamx_')

# data.synchronise_folder()

# ana.score('eve2_', include_metric=True)

# h.val_score(model_name='m_eve_opt', use_metric=True, iter=10, th=0.8)
# h.val_score(model_name='m_eve3_4', use_metric=True, iter=10, th=0.8)
# h.test(model_name='m_eve3_4')
# h.test(model_name='ezekiel_opt')

# h.val_score('evex_484_0.00216_1_11.87_235', use_metric=True, th=0.8)
h.test('adamy_784_0.00269_4_9.57_174_17')



# ana.save_images('adamy', postproc=True)

# ana.plot_analysis('eve2_114_0.3_25', th=0)



# ana.analysis(analysis='lr_ep_bs', analysis_name='adamy', use_metric=True)
# ana.analysis(analysis='ed_ma_sc', analysis_name='evey', use_metric=True)
# ana.analysis(analysis='ed_ma', analysis_name='m_azrael2_', use_metric=True)
# ana.analysis(analysis='ss', analysis_name='evey', use_metric=True)
# ana.analysis(analysis='lr', analysis_name='m_adam4_', use_metric=True)

# ana.score('adamy', include_metric=True, iter=3)

# ana.score('cain', include_metric=True)
# ana.score('adam3')
#
# ana.score('azrael3_')
# ana.score('eve3_')
# ana.score('ezekiel3_')

