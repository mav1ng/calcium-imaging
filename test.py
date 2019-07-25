import torch
import network as n
import data
import corr
import config as c
import helpers as h
import numpy as np
import matplotlib.pyplot as plt
import clustering as cl
import visualization as v
from sklearn.datasets import make_blobs
from torchvision import transforms, utils
import os
from scipy.misc import imread
from glob import glob
from torch import matmul as mm
#
from numpy import array, zeros
import scipy.ndimage as ndimage
import cv2

batch_size = c.training['batch_size']
train = c.training['train']

dtype = c.data['dtype']

device = torch.device('cpu')
img_size = c.training['img_size']



# a = data.load_numpy_from_h5py('plot_data/test_optimal_th_f1_list_gzip.hkl').flatten()
# b = data.load_numpy_from_h5py('plot_data/test_optimal_th_th_list_gzip.hkl').flatten()
#
# plt.plot(b, a)
#
# a = data.load_numpy_from_h5py('plot_data/test_optimal_th_f1_list_bp3_con3_gzip.hkl').flatten()
# b = data.load_numpy_from_h5py('plot_data/test_optimal_th_th_list_bp3_con3_gzip.hkl').flatten()
#
# plt.plot(b, a)
#
# a = data.load_numpy_from_h5py('plot_data/test_optimal_th_f1_list_full2_con_gzip.hkl').flatten()
# b = data.load_numpy_from_h5py('plot_data/test_optimal_th_th_list_full2_con_gzip.hkl').flatten()
#
# plt.plot(b, a)
#
# a = data.load_numpy_from_h5py('plot_data/test_optimal_th_f1_list_full3_gzip.hkl').flatten()
# b = data.load_numpy_from_h5py('plot_data/test_optimal_th_th_list_full3_gzip.hkl').flatten()
#
# plt.plot(b, a)
#
# plt.show()
#
# print(a.shape)
# print(b.shape)
#
# h.val_score(model_name='pms3')
# h.test(model_name='pms3', th=0.75)

# data.preprocess_corr(corr_path='data/corr/starmy/sliced/slice_size_50', nb_corr_to_preserve=4,
#                     use_denoiser=False, test=False)

# a, b = h.test_th(np_arange=(0.01, 2.0, 0.01), iter=10, model_name='full3')
# print(a.shape)
# print(b.shape)
# data.save_numpy_to_h5py(a, 'test_optimal_th_f1_list_full3', 'plot_data/', label_array=None,
#                         use_compression=c.data['use_compression'])
# data.save_numpy_to_h5py(b, 'test_optimal_th_th_list_full3', 'plot_data/', label_array=None,
#                         use_compression=c.data['use_compression'])
#
#
# a, b = h.test_th(np_arange=(0.01, 2.0, 0.01), iter=10, model_name='full2_con')
# print(a.shape)
# print(b.shape)
# data.save_numpy_to_h5py(a, 'test_optimal_th_f1_list_full2_con', 'plot_data/', label_array=None,
#                         use_compression=c.data['use_compression'])
# data.save_numpy_to_h5py(b, 'test_optimal_th_th_list_full2_con', 'plot_data/', label_array=None,
#                         use_compression=c.data['use_compression'])
#
#
# a, b = h.test_th(np_arange=(0.01, 2.0, 0.01), iter=10, model_name='bp3_con3')
# print(a.shape)
# print(b.shape)
# data.save_numpy_to_h5py(a, 'test_optimal_th_f1_list_bp3_con3', 'plot_data/', label_array=None,
#                         use_compression=c.data['use_compression'])
# data.save_numpy_to_h5py(b, 'test_optimal_th_th_list_bp3_con3', 'plot_data/', label_array=None,
#                         use_compression=c.data['use_compression'])

# h.val_score(model_name='full2', th=0.75)
# h.val_score(model_name='full3', th=0.75)
# h.val_score(model_name='full2_con', th=0.75)
# h.val_score(model_name='full2_con2', th=0.75)
# h.val_score(model_name='bp3_con3', th=0.75)
#
# h.val_score(model_name='full2', th=1.95)
# h.val_score(model_name='full3', th=1.95)
# h.val_score(model_name='full2_con', th=1.95)
# h.val_score(model_name='full2_con2', th=1.95)
# h.val_score(model_name='bp3_con3', th=1.95)

# data.get_summary_img(nf_folder='data/test_data', sum_folder='data/test_sum_img',
#                         test=True, device=device, dtype=torch.double)
# data.preprocess_corr(corr_path='data/test_corr/starmy/sliced/slice_size_50', nb_corr_to_preserve=4,
#                     use_denoiser=False, test=True)

# a, b = h.test_th(np_arange=(0.005, 0.015, 0.005), iter=2)
# print(a.shape)
# print(b.shape)
# data.save_numpy_to_h5py(a, 'test_optimal_th_f1_list', 'plot_data/', label_array=None,
#                         use_compression=c.data['use_compression'])
# data.save_numpy_to_h5py(b, 'test_optimal_th_th_list', 'plot_data/', label_array=None,
#                         use_compression=c.data['use_compression'])

# data.generate_data(nf_folder='data/training_data', corr_path='data/corr/starmy/sliced/slice_size_50', slicing=True,
#                     slice_size=50, nb_corr_to_preserve=4, generate_summary=True, sum_folder='data/sum_img',
#                     testset=False, use_denoiser=False)

#
# device = c.cuda['device']
# model = n.UNetMS(background_pred=c.UNet['background_pred'])
#
# model.to(device)

# h.val_score(10, 0.75, 'full2', True)
# h.val_score(10, 0.75, 'bp2', True)
# h.val_score(10, 0.75, 'bp3_con', True)


# a, b, = h.test_th()
# data.save_numpy_to_h5py(a, 'test_optimal_th_f1_list', 'plot_data/', label_array=None,
#                         use_compression=c.data['use_compression'])
# data.save_numpy_to_h5py(b, 'test_optimal_th_th_list', 'plot_data/', label_array=None,
#                         use_compression=c.data['use_compression'])

# data.get_summarized_masks('data/masks')
# masks = data.load_numpy_from_h5py('data/sum_masks/nf_00.09_gzip.hkl')
# print(masks.shape)
# plt.imshow(masks)
# plt.show()

# data.preprocess_corr(corr_path='data/corr/starmy/sliced/slice_size_100/', nb_corr_to_preserve=4, use_denoiser=False)
# data.preprocess_corr(corr_path='data/corr/starmy/sliced/slice_size_100/', nb_corr_to_preserve=8, use_denoiser=False)
# data.preprocess_corr(corr_path='data/corr/suit/sliced/slice_size_100/', nb_corr_to_preserve=4, use_denoiser=False)
# data.preprocess_corr(corr_path='data/corr/suit/sliced/slice_size_100/', nb_corr_to_preserve=8, use_denoiser=False)



# transform = transforms.Compose([data.CorrRandomCrop(img_size, nb_excluded=2, corr_form='starmy')])
# comb_dataset = data.CombinedDataset(corr_path='data/corr/starmy/sliced/slice_size_100/', sum_folder='data/sum_img/',
#                                     transform=None, device=device, dtype=dtype)
#
# input = comb_dataset[11]['image']
# label = comb_dataset[11]['label']
#
# dat = input
# l = label.cpu().numpy()
#
# (c, w, h) = input.size()
#
# # for i in range(c):
# #     d = dat[i].cpu().numpy()
# #
# #     f, axarr = plt.subplots(2)
# #     axarr[0].imshow(d)
# #     axarr[1].imshow(l)
# #
# #     plt.title('Actual Mean (upper) vs Ground Truth (lower)')
# #
# #     plt.show()
#
#
#
# corr_mean = torch.mean(input, dim=0)
# corrected_img = torch.mean(input[2:7], dim=0) - corr_mean
# corrected_img_2 = torch.where(corrected_img < 0., torch.tensor(0.).cpu(), corrected_img)
# print(corrected_img_2.cpu().numpy().astype('uint8'))
#
#
# filtered = denoise(corrected_img_2.cpu().numpy(), weight=0.05, eps=0.00001)
# print(filtered)
#
# f, axarr = plt.subplots(4, 2)
# axarr[0, 0].imshow(input[2].cpu().numpy())
# axarr[0, 1].imshow(corr_mean.cpu().numpy())
# axarr[1, 0].imshow(corrected_img.cpu().numpy())
# axarr[1, 1].imshow(corrected_img_2.cpu().numpy())
# axarr[2, 0].imshow(filtered)
# axarr[2, 1].imshow(filtered)
# axarr[3, 0].imshow(label.cpu().numpy())
# axarr[3, 1].imshow(filtered)
# plt.show()



# train = c.training['train']
# dtype = torch.float
# batch_size = c.training['batch_size']
#
# device = torch.device('cpu')
# img_size = c.training['img_size']
#
# data.get_summary_img('data/training_data/', device=device, dtype=torch.float)




# transform = transforms.Compose([data.CorrRandomCrop(img_size, nb_excluded=2, corr_form='suit')])
# comb_dataset = data.CombinedDataset(corr_path='data/corr/suit/sliced/slice_size_100/', sum_folder='data/sum_img/',
#                                     transform=None, device=device, dtype=dtype)
#
# files = sorted(glob('data/training_data/neurofinder.00.09' + '/images/*.tiff'))
# imgs = torch.tensor(array([imread(f) for f in files]).astype(np.float64), dtype=dtype,
#                     device=device)
#
# mean_summar = data.get_mean_img(imgs)
# h = imgs - mean_summar
# h_ = data.get_mean_img(torch.where(h < 0., torch.tensor(0.), h))
# h = data.get_mean_img(h)
#
# #mean_summar = data.normalize_summary_img(mean_summar)
#
#
# var_summar = data.get_var_img(imgs)
# g = var_summar
# g_ = data.get_var_img(torch.where(g < 0., torch.tensor(0.), g))
# g_ = torch.sqrt(g)
#
# #var_summar = data.normalize_summary_img(var_summar)
#
#
#
# d = mean_summar.cpu().numpy()
# l = var_summar.cpu().numpy()
# h = h.cpu().numpy()
# h_ = h_.cpu().numpy()
# f, axarr = plt.subplots(3, 2)
# axarr[0, 0].imshow(d)
# axarr[0, 1].imshow(l)
# axarr[1, 0].imshow(h)
# axarr[1, 1].imshow(h_)
# axarr[2, 0].imshow(g)
# axarr[2, 1].imshow(g_)
#
#
# plt.title('Actual Mean (upper) vs Ground Truth (lower)')
#
# plt.show()