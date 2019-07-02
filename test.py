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

from torch import matmul as mm

transform = transforms.Compose([data.CorrRandomCrop(c.training['img_size'], nb_excluded=2, corr_form='small_star')])
croppeddataset = data.SingleCombinedDataset(corr_path='data/corr/small_star/sliced/slice_size_100/', nb_dataset=0,
                                            sum_folder='data/sum_img/',
                                            device=c.cuda['device'], dtype=c.data['dtype'], transform=transform)
print('here we are')
print(croppeddataset[0]['image'].size())

# pre = make_blobs(n_samples=1000, n_features=2, centers=None, cluster_std=1.0, center_box=(-10.0, 10.0),
#                  shuffle=True, random_state=None)[0]
# print(pre.shape)
# a = torch.tensor(pre, device=torch.device('cuda'))
# a = a.reshape(1, 1, 2, 1000)
#
# def plot(data):
#     data = data.cpu().numpy()
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#
#     ax.scatter(x=data[:, 0], y=data[:, 1], alpha=0.8, edgecolors='none', s=30)
#
#     plt.legend(loc=2)
#     plt.show()
#
# def forward(x_in):
#     """
#     :param x_in: flattened image in D x N , D embedding Dimension, N number of Pixels
#     :return: tensor with dimension B x D x W x H, B batch size, D embedding dimension, loss
#     W width of the image, H height of the image, embeddings x_in mean shifted
#     """
#
#     iter = 5
#     step_size = 0.5
#     kernel_bandwidth = 1. / (1. - c.embedding_loss['margin']) / 3.
#
#     with torch.no_grad():
#         bs = x_in.size(0)
#         emb = x_in.size(1)
#         w = x_in.size(2)
#         h = x_in.size(3)
#
#     x = x_in.view(bs, emb, -1)
#
#     y = torch.zeros(emb, w * h).cuda()
#     out = torch.zeros(bs, emb, w, h).cuda()
#
#     for t in range(iter + 1):
#         # iterating over all samples in the batch
#         for b in range(bs):
#             y = x[b, :, :]
#
#             if t != 0:
#                 kernel_mat = torch.exp(
#                     torch.mul(kernel_bandwidth, mm(y.clone().t(), y.clone())))
#                 diag_mat = torch.diag(mm(kernel_mat.t(), torch.ones(w * h, 1).cuda())[:, 0], diagonal=0)
#
#                 y = mm(y.clone(),
#                        torch.add(torch.mul(step_size, mm(kernel_mat, torch.inverse(diag_mat))),
#                                  torch.mul(1. - step_size, torch.eye(w * h).cuda())))
#
#             out[b, :, :, :] = y.view(emb, w, h)
#
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
#
