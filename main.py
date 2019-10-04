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

# ana.full_analysis(analysis='ss', analysis_name='ss_noah_', plot_name='ss_noah', figsize=(10, 3))
# ana.full_analysis(analysis='scaling_comb', analysis_name='scale_noah_', plot_name='sc_combined', figsize=(10, 4))
# ana.full_analysis(analysis='m_emb', analysis_name='m_emb_noah_', plot_name='m_emb_noah', figsize=(15, 11))
# ana.full_analysis(analysis='iter', analysis_name='iter_noah_', plot_name='iter_noah', figsize=(5, 5))
# ana.full_analysis(analysis='kb', analysis_name='kb_noah_', plot_name='kb_noah', figsize=(5, 5))

# h.test('kb_noah_25.0', cl_th=1.5, pp_th=0.2, obj_size=20, hole_size=20, show_image=True, save_image=False)


# v.plot_learning_curve(model_name='azrael', cutoff_1=20, cutoff_2=20, figsize=(10, 4), azrael=False)
# v.plot_iter_curve(figsize=(10, 4), cutoff_1=50, cutoff_2=50)

# """NOAH OPT EMB Margin"""
# nb_epochs = 50
# step_size = 1.
#
# subsample_size = 1024
# margin_list = np.linspace(0.1, 0.9, 1000)
#
# for i in range(20):
#     margin = np.around(np.random.choice(margin_list), decimals=2)
#     emb_dim = np.random.randint(2, 65)
#     nb_iter = 1
#     kernel_bandwidth = 2
#     scaling = 4.
#     bs = 20
#     lr = 0.0002
#
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin: ', margin, 'Scaling: ', scaling,
#           'Number epochs: ',
#           nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs, 'kernel_bandwidth', kernel_bandwidth)
#     set = h.Setup(
#         model_name='m_emb_noah_' + str(margin) + '_' + str(emb_dim),
#         subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, scaling=scaling,
#         nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=False,
#         background_pred=True,
#         nb_iterations=nb_iter, kernel_bandwidth=kernel_bandwidth, step_size=step_size, embedding_loss=True)
#     set.main()
# ana.score('m_emb_noah_', include_metric=True)
# ana.save_images('m_emb_noah_')

# h.test('3D_emb_50', cl_th=0.75, pp_th=0.2, hole_size=14, obj_size=20)
# h.test('3D_emb_init', cl_th=0.75, pp_th=0.2, hole_size=14, obj_size=20)


# h.test('noah_opt_75', cl_th=1.5, pp_th=0.2, obj_size=20, hole_size=20, show_image=True, save_image=False)

# ana.full_score('m_emb_azrael_', include_metric=False, iter=10)
# ana.full_score('pre_adam_trained2', include_metric=True, iter=10)
# ana.full_score('pre_eve_trained2', include_metric=True, iter=10)
# ana.full_score('m_emb_azrael_0.89 47', include_metric=True, iter=10)
# ana.full_score('m_emb_noah_', include_metric=True, iter=10)

# data.synchronise_folder()

# a = h.find_th('abram_opt_30', iter=10)

# th_cl, th_pp = h.find_th('noah_opt_225')
# best_obj, best_hole = h.find_optimal_object_size('noah_opt_225', cl_th=th_cl, pp_th=th_pp)
# h.val_score('noah_opt_75', use_metric=True, iter=100, cl_th=1.5, pp_th=0.175, return_full=True, obj_size=20,
#              holes_size=20)

# h.find_optimal_object_size('abram_opt_30', cl_th=.1, pp_th=0.4)
# h.find_optimal_object_size('azrael_opt_112', cl_th=.1, pp_th=0.25)
# h.find_optimal_object_size('eve_opt_132', cl_th=1.25, pp_th=0.375)
# h.find_optimal_object_size('adam_opt_132', cl_th=.75, pp_th=0.15)

# h.create_output_image('abram_opt_30', cl_th=.1, pp_th=0.4, obj_size=10, hole_size=12, show_image=False, save_images=True)
# h.create_output_image('azrael_opt_112', cl_th=.1, pp_th=0.25, obj_size=35, hole_size=6, show_image=False, save_images=True)
# h.create_output_image('eve_opt_132', cl_th=1.25, pp_th=0.25, obj_size=10, hole_size=16, show_image=True, save_images=False)
# h.create_output_image('adam_opt_132', cl_th=.75, pp_th=0.15, obj_size=20, hole_size=14, show_image=False, save_images=True)
# h.create_output_image('noah_opt_75', cl_th=1.5, pp_th=0.175, obj_size=20, hole_size=20, show_image=False, save_images=True)

# h.test('abram_opt_30', cl_th=.1, pp_th=0.4, obj_size=10, hole_size=12, show_image=False, save_image=False)
# h.test('noah4_0.0002_1_16_2')
# h.test('azrael_opt_112', cl_th=0.1, pp_th=0.25, obj_size=35, hole_size=6, show_image=False, save_image=False)
# h.test('adam_opt_132', cl_th=0.75, pp_th=0.15, obj_size=20, hole_size=14, show_image=False, save_image=False)
# h.test('eve_opt_132', cl_th=1.25, pp_th=0.25, obj_size=10, hole_size=16, show_image=False, save_image=False)

# a = h.find_th('abram_opt_30', iter=10)

# h.val_score('abram_opt_30', cl_th=.1, pp_th=0.4, obj_size=10, holes_size=12, use_metric=True, iter=100, return_full=True)
# h.val_score('azrael_opt_112', cl_th=.1, pp_th=0.25, obj_size=35, holes_size=6, use_metric=True, iter=100, return_full=True)
# h.val_score('eve_opt_132', cl_th=1.25, pp_th=0.25, use_metric=True, obj_size=10, holes_size=16, iter=100, return_full=True)
# h.val_score('adam_opt_132', cl_th=.75, pp_th=0.15, obj_size=20, holes_size=14, use_metric=True, iter=100, return_full=True)
# h.val_score('noah4_0.0002_1_16_2', cl_th=0.75, pp_th=0.125, obj_size=20, holes_size=14, use_metric=True, iter=100, return_full=True)



# for i in range(19):
#     input_test = ana.input_test(nb_neuro=i, input_dim=0, corr_path='data/corr/starmy/maxpool/transformed_4/',
#                    corr_sum_folder='data/corr_sum_img/',
#                    sum_folder='data/sum_img/', show_label=False)
#     input_test_2 = ana.input_test(nb_neuro=i, input_dim=0, corr_path='data/corr/starmy/maxpool/transformed_4/',
#                    corr_sum_folder='data/corr_sum_img/',
#                    sum_folder='data/sum_img/', show_label=True)
#     ana.show_input(input_test, 'input_' + str(i), save_image=True)
#     ana.show_input(input_test_2, 'labels_' + str(i), save_image=True)


# for i in [2, 3, 6, 10, 20]:
#     input_test = ana.input_test(nb_neuro=1, input_dim=i, corr_path='data/x_mp_corr_folder_no_pre/',
#                        corr_sum_folder='data/corr_sum_img/',
#                        sum_folder='data/x_mp_sum_folder_no_pre/')
#     ana.show_input(input_test, str(i) + '_corrs_0001', save_image=False)



# ana.score('noah3_', include_metric=False)
# ana.save_images('noah3_')


# h.val_score('abram_0.03426_4', use_metric=True, iter=3, th=)

# h.find_th('abram_opt', iter=10)

# h.test('adam__0.00039_3.9_10_19')
# h.test('adam__0.00027_4.0_16_20')
# val_dataset = data.CombinedDataset(corr_path='data/corr/starmy/maxpool/transformed_4/',
#                                    corr_sum_folder='data/corr_sum_img/',
#                                    sum_folder='data/sum_img/',
#                                    mask_folder='data/sum_masks/',
#                                    transform=None, device=device, dtype=dtype, test=True)

# data.generate_data(nf_folder='data/training_data', slice_size=100, corr_path='data/x_no_ms_corr_folder', slicing=True, sum_folder='data/x_no_ms_sum_folder'
#                    , nb_corr_to_preserve=4, generate_summary=True, maxpool=True, preprocess=True)

# data.get_summary_img(nf_folder='data/test_data', sum_folder='data/test_sum_img',
#                         test=True, device=torch.device('cpu'), dtype=torch.double, maxpool=True)


# """NOAH OPT ROUND 3"""
# margin = 0.5
# nb_epochs = 50
# nb_iter = 1
# step_size = 1.
# emb_dim = 32
#
#
# kernel_bandwidth_list = np.linspace(5, 15, 1000)
# lr_list = np.linspace(0.0001, 0.01, 10000)
# subsample_size = 1024
#
# for i in range(50):
#     kernel_bandwidth = np.around(np.random.choice(np.array([3., 6., 10.])), decimals=2)
#     scaling = 4.
#     bs = 1
#     lr = np.around(np.random.choice(lr_list), decimals=5)
#
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin: ', margin, 'Scaling: ', scaling,
#           'Number epochs: ',
#           nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs, 'kernel_bandwidth', kernel_bandwidth)
#     set = h.Setup(
#         model_name='noah3_' + str(lr) + '_' + str(bs) + '_' + str(scaling) + '_' + str(kernel_bandwidth),
#         subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, scaling=scaling,
#         nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=False,
#         background_pred=True,
#         nb_iterations=nb_iter, kernel_bandwidth=kernel_bandwidth, step_size=step_size, embedding_loss=True)
#     set.main()
# ana.score('noah3_', include_metric=True)
# ana.save_images('noah3_')


# h.test('mean_shift_full_test')

# set = h.Setup(
#         model_name='mean_shift_iter1_bs1_kb10',
#         subsample_size=1024, embedding_dim=32, margin=0.5, scaling=1.,
#         nb_epochs=10, save_config=True, learning_rate=0.002, batch_size=1, include_background=False,
#         background_pred=True,
#         nb_iterations=1, kernel_bandwidth=10., step_size=1., embedding_loss=True)
# set.main()

# set = h.Setup(
#         model_name='mean_shift_full_test_0.5',
#         subsample_size=1024, embedding_dim=32, margin=0.5, scaling=3.,
#         nb_epochs=10, save_config=True, learning_rate=0.0005, batch_size=20, include_background=False,
#         background_pred=True,
#         nb_iterations=3, kernel_bandwidth=6., step_size=.5, embedding_loss=True)
# set.main()

# ana.score('noah4', include_metric=False)
# ana.save_images('noah4')

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

# h.test('noah_0.00315_11_7.17_11.27')


# ana.analysis(analysis='lr_ep_bs', analysis_name='noah4', use_metric=False)
# ana.analysis(analysis='ed_ma_sc', analysis_name='noah4', use_metric=False)
# ana.analysis(analysis='ed_ma', analysis_name='azrael2', use_metric=False)
# ana.analysis(analysis='kb_iter', analysis_name='noah4', use_metric=False)
# ana.analysis(analysis='ss', analysis_name='evey', use_metric=True)
# ana.analysis(analysis='lr', analysis_name='m_adam4_', use_metric=True)

# ana.score('noah', include_metric=False, iter=3)

# ana.score('cain', include_metric=True)
# ana.score('adam3')
#
# ana.score('azrael3_')
# ana.score('eve3_')
# ana.score('ezekiel3_')

