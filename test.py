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
from scipy.misc import imread
from glob import glob
from torch import matmul as mm
#
from numpy import array, zeros
import scipy.ndimage as ndimage
import cv2

def denoise(img, weight=0.1, eps=1e-3, num_iter_max=10000):
    """Perform total-variation denoising on a grayscale image.

    Parameters
    ----------
    img : array
        2-D input data to be de-noised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more
        de-noising (at the expense of fidelity to `img`).
    eps : float, optional
        Relative difference of the value of the cost
        function that determines the stop criterion.
        The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    num_iter_max : int, optional
        Maximal number of iterations used for the
        optimization.

    Returns
    -------
    out : array
        De-noised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """

    u = np.zeros_like(img)
    px = np.zeros_like(img)
    py = np.zeros_like(img)

    nm = np.prod(img.shape[:2])
    tau = 0.125

    i = 0
    while i < num_iter_max:
        u_old = u

        # x and y components of u's gradient
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u

        # update the dual variable
        px_new = px + (tau / weight) * ux
        py_new = py + (tau / weight) * uy

        norm_new = np.maximum(1, np.sqrt(px_new ** 2 + py_new ** 2))
        px = px_new / norm_new
        py = py_new / norm_new

        # calculate divergence
        rx = np.roll(px, 1, axis=1)
        ry = np.roll(py, 1, axis=0)
        div_p = (px - rx) + (py - ry)

        # update image
        u = img + weight * div_p

        # calculate error
        error = np.linalg.norm(u - u_old) / np.sqrt(nm)

        if i == 0:
            err_init = error
            err_prev = error
        else:
            # break if error small enough
            if np.abs(err_prev - error) < eps * err_init:
                break
            else:
                e_prev = error

        # don't forget to update iterator
        i += 1
    return u


train = c.training['train']
dtype = c.data['dtype']
batch_size = c.training['batch_size']

device = torch.device('cpu')
img_size = c.training['img_size']

transform = transforms.Compose([data.CorrRandomCrop(img_size, nb_excluded=2, corr_form='starmy')])
comb_dataset = data.CombinedDataset(corr_path='data/corr/starmy/sliced/slice_size_100/', sum_folder='data/sum_img/',
                                    transform=None, device=device, dtype=dtype)

input = comb_dataset[4]['image']
label = comb_dataset[4]['label']

dat = input
l = label.cpu().numpy()

(c, w, h) = input.size()

# for i in range(c):
#     d = dat[i].cpu().numpy()
#
#     f, axarr = plt.subplots(2)
#     axarr[0].imshow(d)
#     axarr[1].imshow(l)
#
#     plt.title('Actual Mean (upper) vs Ground Truth (lower)')
#
#     plt.show()



corr_mean = torch.mean(input, dim=0)
corrected_img = input[3] - corr_mean
corrected_img_2 = torch.where(corrected_img < 0., torch.tensor(0.).cpu(), corrected_img)
print(corrected_img_2.cpu().numpy().astype('uint8'))


filtered = denoise(corrected_img_2.cpu().numpy(), weight=0.05, eps=0.00001)
print(filtered)

f, axarr = plt.subplots(4, 2)
axarr[0, 0].imshow(input[2].cpu().numpy())
axarr[0, 1].imshow(corr_mean.cpu().numpy())
axarr[1, 0].imshow(corrected_img.cpu().numpy())
axarr[1, 1].imshow(corrected_img_2.cpu().numpy())
axarr[2, 0].imshow(filtered)
axarr[2, 1].imshow(filtered)
axarr[3, 0].imshow(label.cpu().numpy())
axarr[3, 1].imshow(filtered)
plt.show()



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