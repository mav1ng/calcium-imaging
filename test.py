import torch
import data
import corr
import config as c
import numpy as np
import matplotlib.pyplot as plt
import clustering as cl
import visualization as v
from sklearn.datasets import make_blobs
from torchvision import transforms, utils
import os

from torch import matmul as mm
#
# transform = transforms.Compose([data.CorrRandomCrop(c.training['img_size'], nb_excluded=2, corr_form='small_star')])
# croppeddataset = data.SingleCombinedDataset(corr_path='data/corr/small_star/sliced/slice_size_100/', nb_dataset=0,
#                                             sum_folder='data/sum_img/',
#                                             device=c.cuda['device'], dtype=c.data['dtype'], transform=transform)
# print('here we are')
# print(croppeddataset[0]['image'].size())

dtype = torch.double
device = torch.device('cpu')

for index, folder in enumerate(sorted(os.listdir('data/training_data'))):
    print(folder)
    corr = data.create_corr_data(neurofinder_path='data/training_data/' + str(folder), slicing=False, corr_form='right',
                                 dtype=dtype, device=device)
    data.save_numpy_to_h5py(data_array=corr['correlations'].numpy(), label_array=corr['labels'].numpy(),
                            file_name='corr_nf_' + str(index), file_path='data/corr/right/full/')

for index, folder in enumerate(sorted(os.listdir('data/training_data'))):
    print(folder)
    corr = data.create_corr_data(neurofinder_path='data/training_data/' + str(folder), slicing=True, slice_size=100,
                                 corr_form='right', dtype=dtype, device=device)
    data.save_numpy_to_h5py(data_array=corr['correlations'].numpy(), label_array=corr['labels'].numpy(),
                            file_name='corr_nf_' + str(index), file_path='data/corr/right/sliced/slice_size_100/')

# pre = make_blobs(n_samples=1000, n_features=2, centers=None, cluster_std=1.0, center_box=(-10.0, 10.0),
#                  shuffle=True, random_state=None)[0]
# print(pre.shape)
# a = torch.tensor(pre, device=torch.device('cuda'))
# a = a.reshape(1, 1, 2, 1000)
# a = torch.rand(1, 3, 2, 1000, device=torch.device('cuda'))
#
#
# def plot(data):
#     data = data.cpu().numpy()
#     print(data)
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#
#     ax.scatter(x=data[:, 0], y=data[:, 1])
#
#     plt.show()
#
#
# def forward(x_in):
#     """
#     :param x_in: flattened image in D x N , D embedding Dimension, N number of Pixels
#     :return: tensor with dimension B x D x W x H, B batch size, D embedding dimension, loss
#     W width of the image, H height of the image, embeddings x_in mean shifted
#     """
#
#     iter = 5
#     step_size = 1.
#     kernel_bandwidth = 6.
#
#     with torch.no_grad():
#         bs = x_in.size(0)
#         emb = x_in.size(1)
#         w = x_in.size(2)
#         h = x_in.size(3)
#
#     x = x_in.view(bs, emb, -1)
#
#     print(x.size())
#
#     for b in range(bs):
#         x_ = torch.mean(x[b], dim=0)
#         print(x_, x[b])
#         print(x_)
#         x_n = x[b] - x_
#         print(x_n.size())
#         x_n_ = torch.sqrt(torch.sum(x_n ** 2, dim=0))
#         print(x_n_)
#         x[b] = x_n / x_n_
#
#
#     y = torch.zeros(emb, w * h).cuda()
#     out = torch.zeros(bs, emb, w, h).cuda()
#
#     for t in range(iter + 1):
#         # iterating over all samples in the batch
#         for b in range(bs):
#             y = x[b, :, :]
#             print('y pre', y)
#             if t != 0:
#                 print(torch.max(mm(y.clone().t(), y.clone())))
#                 kernel_mat = torch.exp(torch.mul(kernel_bandwidth, mm(y.clone().t(), y.clone())))
#                 diag_mat = torch.diag(mm(kernel_mat.t(), torch.ones(w * h, 1).cuda())[:, 0], diagonal=0)
#
#                 print('kernel_mat', kernel_mat)
#                 print('diag_mat', diag_mat)
#
#                 print(torch.mul(step_size, mm(kernel_mat, torch.inverse(diag_mat)))
#                       )
#                 print(torch.mul(1. - step_size, torch.eye(w * h).cuda()))
#                 y = mm(y.clone(),
#                        torch.add(torch.mul(step_size, mm(kernel_mat, torch.inverse(diag_mat))),
#                                  torch.mul(1. - step_size, torch.eye(w * h).cuda())))
#                 print('y', y)
#             out[b, :, :, :] = y.view(emb, w, h)
#             print('out', out)
#         print(out[0].view(2, -1).t().size())
#         plot(out[0].view(2, -1).t())
#
#         x = out.view(bs, emb, -1)
#
#     return out
#
# plot(a[0].view(2, -1).t())
# a_ = a[0].view(2, -1).t()
# b = forward(a)[0].view(2, -1).t()
# plot(forward(a)[0].view(2, -1).t())

