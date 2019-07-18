from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import json
import matplotlib.pyplot as plt
from numpy import array, zeros
import numpy as np
from scipy.misc import imread
from glob import glob
import corr
import hickle as hkl

import config as c
import helpers as h


def tomask(coords, dims):
    """
    Method that return the Mask of the given coordinates of the regions
    :param coords:
    :param dims:
    :return:
    """
    mask = zeros(dims)
    mask[list(zip(*coords))] = 1.
    return mask


def toCoords(mask):
    """
    Method that returns the Coordinates of the Regions in the Mask
    :param mask:
    :return:
    """
    try:
        unique = torch.unique(mask)
    except TypeError:
        mask = torch.tensor(mask).cuda()
        unique = torch.unique(mask)
    coords = []
    for _, label in enumerate(unique):
        coords.append({'coordinates': (mask == label).nonzero().cpu().numpy().tolist()})
    return coords


def write_to_json(data, path):
    """
    Method ro write to json File
    :param data:
    :param path:
    :return:
    """
    with open(path, 'w') as outfile:
        json.dump(data, outfile)
    pass


def get_summarized_masks(masks_path):
    """Default summary function for a mask. Flattens the stack of neuron masks
    into a (height x width) combined mask. Eliminates overlapping and neighboring
    pixels that belong to different neurons to preserve the original number of
    independent neurons.
    # Arguments
        dspath: Path to HDF5 dataset where the mask is stored.
    # Returns
        summ: (height x width) mask summary.
        """

    for index, folder in enumerate(sorted(os.listdir(masks_path))):
        print('Summarizing file ' + str(folder))

        # fp = h5py.File(folder)
        # msks = fp.get('masks/raw')[...]
        # fp.close()

        msks = load_numpy_from_h5py(str(masks_path) + '/' + str(folder))

        # Coordinates of all 1s in the stack of masks.
        zyx = list(zip(*np.where(msks == 1)))

        # Mapping (y,x) -> z.
        yx_z = {(y, x): [] for z, y, x in zyx}
        for z, y, x in zyx:
            yx_z[(y, x)].append(z)

        # Remove all elements with > 1 z.
        for k in list(yx_z.keys()):
            if len(yx_z[k]) > 1:
                del yx_z[k]
        assert np.max([len(v) for v in yx_z.values()]) == 1.

        # For (y,x), take the union of its z-values with its immediate neighbors' z-values.
        # Delete the (y,x) and its neighbors if |union| > 1.
        for y, x in list(yx_z.keys()):
            nbrs = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1), (y + 1, x + 1),
                    (y - 1, x - 1), (y + 1, x - 1), (y - 1, x + 1)] + [(y, x)]
            nbrs = [k for k in nbrs if k in yx_z]
            allz = [yx_z[k][0] for k in nbrs]
            if len(np.unique(allz)) > 1:
                for k in nbrs:
                    del yx_z[k]

        # The mask consists of the remaining (y,x) keys.
        yy, xx = [y for y, x in yx_z.keys()], [x for y, x in yx_z.keys()]
        summ = np.zeros(msks.shape[1:])
        summ[yy, xx] = 1.

        save_numpy_to_h5py(data_array=summ, file_name=folder[:8], file_path='data/sum_masks/')



def get_masks(nf_path):

    for index, folder in enumerate(sorted(os.listdir(nf_path))):

        path = str(nf_path) + '/' + str(folder)

        files = sorted(glob(path + '/images/*.tiff'))
        imgs = array([imread(f) for f in files])
        dims = imgs.shape[1:]  # 512 x 512

        with open(path + '/regions/regions.json') as f:
            regions = json.load(f)

        mask = array([tomask(s['coordinates'], dims) for s in regions])

        save_numpy_to_h5py(data_array=mask, file_name='nf_' + str(path[-5:]), file_path='data/masks/')


class LabelledDataset(Dataset):
    """Labelled Dataset with Correlations and Mean Summary image and Var Summary image"""

    def __init__(self, corr_path, sum_folder='data/sum_img/', mask_folder='data/sum_masks/', transform=None, test=False,
                 dtype=c.data['dtype'],
                 device=c.cuda['device']):
        """
        :param folder_path: Path to the Folder with h5py files with Numpy Array of Correlation Data that should
        be used for training/testing
        :param transform: whether a transform should be used on a sample that is getting drawn
        """

        self.folder_path = corr_path
        self.transform = transform
        self.files = sorted(glob(corr_path + '*.hkl'))
        self.masks = sorted(glob(mask_folder + '*.hkl'))
        self.sum_img = sorted(glob(sum_folder + '*.hkl'))
        self.imgs = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.files if 'labels' not in f and '16' not in f], dtype=dtype,
            device=device)
        self.labels = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.masks if '03.00' not in f], dtype=dtype,
            device=device)
        self.dims = self.imgs.shape[2:]  # 512 x 512
        self.len = self.imgs.shape[0]
        self.test = test
        self.dtype = dtype
        self.sum_mean = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.sum_img if 'var' not in f and '03.00' not in f],
            dtype=dtype,
            device=device)
        self.sum_var = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.sum_img if 'mean' not in f and '03.00' not in f],
            dtype=dtype,
            device=device)
        self.train_val_ratio = 0.75
        self.x_bound = int(round(self.train_val_ratio * self.dims[0]))
        self.y_bound = int(round(self.train_val_ratio * self.dims[1]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if not self.test:
            sample = {'image': torch.cat([self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx],
                                          self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx],
                                          self.labels[idx].view(1, self.dims[0], self.dims[1]),
                                          self.imgs[idx]], dim=0)[
                               :, :self.x_bound, :self.y_bound],
                      'label': self.labels[idx][:self.x_bound, :self.y_bound]}
            # markdown
        else:
            sample = {'image': torch.cat([self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx],
                                          self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx],
                                          self.labels[idx].view(1, self.dims[0], self.dims[1]),
                                          self.imgs[idx]], dim=0)[
                               :, self.x_bound:, self.y_bound:],
                      'label': self.labels[idx][self.x_bound:, self.y_bound:]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class SingleCombinedDataset(Dataset):
    """Combined Dataset with Correlations and Mean Summary image and Var Summary image"""

    def __init__(self, corr_path, nb_dataset, sum_folder='data/sum_img/', mask_folder='data/sum_masks/', transform=None,
                 test=False, dtype=c.data['dtype'],
                 device=c.cuda['device']):
        """
        :param folder_path: Path to the Folder with h5py files with Numpy Array of Correlation Data that should
        be used for training/testing
        :param transform: whether a transform should be used on a sample that is getting drawn
        """

        self.folder_path = corr_path
        self.transform = transform
        self.files = sorted(glob(corr_path + '*.hkl'))
        self.masks = sorted(glob(mask_folder + '*.hkl'))
        self.sum_img = sorted(glob(sum_folder + '*.hkl'))
        self.imgs = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.files if 'labels' not in f and '16' not in f], dtype=dtype,
            device=device)
        self.labels = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.masks if '03.00' not in f], dtype=dtype,
            device=device)
        self.dims = self.imgs.shape[2:]  # 512 x 512
        self.len = self.imgs.shape[0]
        self.idx = nb_dataset
        self.len = 1
        self.test = test
        self.dtype = dtype
        self.sum_mean = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.sum_img if 'var' not in f and '03.00' not in f],
            dtype=dtype,
            device=device)
        self.sum_var = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.sum_img if 'mean' not in f and '03.00' not in f],
            dtype=dtype,
            device=device)
        self.train_val_ratio = 0.75
        self.x_bound = int(round(self.train_val_ratio * self.dims[0]))
        self.y_bound = int(round(self.train_val_ratio * self.dims[1]))

        self.sum_mean = self.sum_mean[self.idx]
        self.sum_var = self.sum_var[self.idx]
        self.imgs = self.imgs[self.idx]
        self.labels = self.labels[self.idx]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if not self.test:
            sample = {'image': torch.cat([self.sum_mean.view(1, self.dims[0], self.dims[1]),
                                          self.sum_var.view(1, self.dims[0], self.dims[1]),
                                          self.imgs], dim=0)[:, :self.x_bound, :self.y_bound],
                      'label': self.labels[:self.x_bound, :self.y_bound]}
            # markdown
        else:
            sample = {'image': torch.cat([self.sum_mean.view(1, self.dims[0], self.dims[1]),
                                          self.sum_var.view(1, self.dims[0], self.dims[1]),
                                          self.imgs], dim=0)[:, self.x_bound:, self.y_bound:],
                      'label': self.labels[self.x_bound:, self.y_bound:]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class TestCombinedDataset(Dataset):
    """Test Combined Dataset with Correlations and Mean Summary image and Var Summary image"""

    def __init__(self, corr_path, sum_folder='data/test_sum_img', transform=None,
                 dtype=c.data['dtype'],
                 device=c.cuda['device']):
        """
        :param folder_path: Path to the Folder with h5py files with Numpy Array of Correlation Data that should
        be used for testing
        :param transform: whether a transform should be used on a sample that is getting drawn
        """

        self.corr_path = str(corr_path) + '/'
        self.sum_folder = str(sum_folder) + '/'
        self.folder_path = self.corr_path
        self.files = sorted(glob(self.corr_path + '*.hkl'))
        self.sum_img = sorted(glob(self.sum_folder + '*.hkl'))
        self.transform = transform

        self.imgs = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.files if 'labels' not in f], dtype=dtype,
            device=device)
        self.dims = self.imgs.shape[2:]  # 512 x 512
        self.len = self.imgs.shape[0]
        self.dtype = dtype
        self.sum_mean = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.sum_img if 'var' not in f],
            dtype=dtype,
            device=device)
        self.sum_var = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.sum_img if 'mean' not in f],
            dtype=dtype,
            device=device)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {'image': torch.cat([self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx],
                                      self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx],
                                      self.imgs[idx]], dim=0)[:, :, :]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class CombinedDataset(Dataset):
    """Combined Dataset with Correlations and Mean Summary image and Var Summary image"""

    def __init__(self, corr_path, sum_folder='data/sum_img/', mask_folder='data/sum_masks/', transform=None, test=False,
                 dtype=c.data['dtype'],
                 device=c.cuda['device']):
        """
        :param folder_path: Path to the Folder with h5py files with Numpy Array of Correlation Data that should
        be used for training/testing
        :param transform: whether a transform should be used on a sample that is getting drawn
        """

        self.folder_path = corr_path
        self.transform = transform
        self.files = sorted(glob(corr_path + '*.hkl'))
        self.masks = sorted(glob(mask_folder + '*.hkl'))
        self.sum_img = sorted(glob(sum_folder + '*.hkl'))

        self.imgs = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.files if 'labels' not in f and '16' not in f], dtype=dtype,
            device=device)
        self.labels = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.masks if '03.00' not in f], dtype=dtype,
            device=device)
        self.dims = self.imgs.shape[2:]  # 512 x 512
        self.len = self.imgs.shape[0]
        self.test = test
        self.dtype = dtype
        self.sum_mean = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.sum_img if 'var' not in f and '03.00' not in f],
            dtype=dtype,
            device=device)
        self.sum_var = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.sum_img if 'mean' not in f and '03.00' not in f],
            dtype=dtype,
            device=device)
        self.train_val_ratio = 0.75
        # self.x_bound = int(round(self.train_val_ratio * self.dims[0]))
        self.y_bound = int(round(self.train_val_ratio * self.dims[1]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if not self.test:
            sample = {'image': torch.cat([self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx],
                                          self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx],
                                          self.imgs[idx]], dim=0)[:, :, :self.y_bound],
                      'label': self.labels[idx][:, :self.y_bound]}
            # markdown
        else:
            sample = {'image': torch.cat([self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx],
                                          self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx],
                                          self.imgs[idx]], dim=0)[:, :, self.y_bound:],
                      'label': self.labels[idx][:, self.y_bound:]}
            # print(sample['image'].size())
            # print(sample['label'].size())
        if self.transform:
            sample = self.transform(sample)
        return sample


# class SummaryDataset(Dataset): -- NOT WORKING RIGHT NOW RETURN WITH THE CONCATENATE IS WRONG
#     """Summary Dataset"""
#
#     def __init__(self, corr_path, sum_folder, transform=None, test=False, dtype=c.data['dtype'], var=False,
#                  device=c.cuda['device']):
#         """
#         :param folder_path: Path to the Folder with h5py files with Numpy Array of Correlation Data that should
#         be used for training/testing
#         :param transform: whether a transform should be used on a sample that is getting drawn
#         """
#
#         self.folder_path = corr_path
#         self.transform = transform
#         self.files = sorted(glob(corr_path + '*.hkl'))
#         self.sum_img = sorted(glob(sum_folder + '*.hkl'))
#         self.labels = torch.tensor(
#             [load_numpy_from_h5py(file_name=f) for f in self.files if 'labels' in f and '16' not in f], dtype=dtype,
#             device=device)
#         self.dims = self.labels.shape[1:]  # 512 x 512
#         self.len = self.labels.shape[0]
#         self.test = test
#         self.var = var
#         self.dtype = dtype
#         self.sum_mean = torch.tensor(
#             [load_numpy_from_h5py(file_name=f) for f in self.sum_img if 'var' not in f and '03.00' not in f],
#             dtype=dtype,
#             device=device)
#         self.sum_var = torch.tensor(
#             [load_numpy_from_h5py(file_name=f) for f in self.sum_img if 'mean' not in f and '03.00' not in f],
#             dtype=dtype,
#             device=device)
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, idx):
#         if not self.test:
#             if self.var:
#                 sample = {'image': torch.cat((self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx],
#                                               self.sum_var.view(1, -1, self.dims[0], self.dims[1])[:, idx])),
#                           'label': self.labels[idx]}
#             else:
#                 sample = {'image': self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx],
#                           'label': self.labels[idx]}
#         else:
#             if self.var:
#                 sample = {'image': torch.cat((self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx],
#                                               self.sum_var.view(1, -1, self.dims[0], self.dims[1])[:, idx]))}
#             else:
#                 sample = {'image': self.sum_mean.view(1, -1, self.dims[0], self.dims[1])[:, idx]}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample




class CorrelationDataset(Dataset):
    """Correlation Dataset"""

    def __init__(self, folder_path, mask_folder='data/sum_masks/', transform=None, test=False, dtype=c.data['dtype'], device=c.cuda['device']):
        """
        :param folder_path: Path to the Folder with h5py files with Numpy Array of Correlation Data that should
        be used for training/testing
        :param transform: whether a transform should be used on a sample that is getting drawn
        """

        self.folder_path = folder_path
        self.transform = transform
        self.files = sorted(glob(folder_path + '*.hkl'))
        self.masks = sorted(glob(mask_folder + '*.hkl'))
        self.imgs = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.files if 'labels' not in f and '16' not in f], dtype=dtype,
            device=device)
        self.labels = torch.tensor(
            [load_numpy_from_h5py(file_name=f) for f in self.masks if '03.00' not in f], dtype=dtype,
            device=device)
        self.dims = self.imgs.shape[2:]  # 512 x 512
        self.len = self.imgs.shape[0]
        self.test = test
        self.dtype = dtype

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if not self.test:
            sample = {'image': self.imgs[idx, :, :, :],
                      'label': self.labels[idx, :, :]}
        else:
            sample = {'image': self.imgs[idx, :, :, :]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class NeurofinderDataset(Dataset):
    """Neurofinder Dataset"""

    def __init__(self, neurofinder_path, transform=None, test=False):
        """
        :param neurofinder_path: Path to the neurofinder dataset
        :param transform: whether a transform should be used on a sample that is getting drawn
        """

        self.neurofinder_path = neurofinder_path
        self.files = sorted(glob(neurofinder_path + '/images/*.tiff'))
        self.imgs = array([imread(f) for f in self.files])
        self.dims = self.imgs.shape[1:]  # 512 x 512
        self.len = self.imgs.shape[0]  # 3024
        self.transform = transform
        self.different_labels = c.data['different_labels']
        self.test = test

        # load the regions (training data only)
        if not test:
            with open(neurofinder_path + '/regions/regions.json') as f:
                self.regions = json.load(f)

            self.mask = array([tomask(s['coordinates'], self.dims) for s in self.regions])
            self.counter = 0

            if self.different_labels:
                for s in self.mask:
                    self.mask[self.counter, :, :] = np.where(s == 1., 1. + self.counter, 0.)
                    self.counter = self.counter + 1

            self.mask = np.amax(self.mask, axis=0)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if not self.test:
            sample = {'image': self.imgs[idx, :, :], 'label': self.mask}
        else:
            sample = {'image': self.imgs[idx, :, :]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomCrop(object):
    """Copied from Pytorch Documentation


    Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, val=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.val = val

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[1:3]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        label_ = label[top: top + new_h, left: left + new_w]

        # ignoring samples where neuron density too low
        while (label_.nonzero().size(0)) / (new_h * new_w) <= c.training['min_neuron_pixels']:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            label_ = label[top: top + new_h, left: left + new_w]
            if self.val:
                break

        label = label_
        image = image[:, top: top + new_h, left: left + new_w]

        return {'image': image, 'label': label}


class CorrRandomCrop(object):
    """Copied from Pytorch Documentation

    Crop with Correlation Correction
    Crop randomly the correlation image in a sample with regards to the calculated correlations.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, nb_excluded, corr_form=c.corr['corr_form'], device=c.cuda['device'],
                 dtype=c.data['dtype']):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.corr_form = corr_form
        self.device = device
        self.dtype = dtype
        self.exlude = nb_excluded

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        ch, h, w = image.shape[:3]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        label_ = label[top: top + new_h, left: left + new_w]

        # ignoring samples where neuron density too low
        while (label_.nonzero().size(0)) / (new_h * new_w) <= c.training['min_neuron_pixels']:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            label_ = label[top: top + new_h, left: left + new_w]

        label = label_
        image = image[:, top: top + new_h, left: left + new_w]

        # deleting information about not available offset pixels, need 2 dimensions otherwise correlation always 0
        correction_image = corr.get_corr(
            torch.rand(2, self.output_size[0], self.output_size[1], device=self.device, dtype=self.dtype),
            self.corr_form, device=self.device, dtype=self.dtype)
        image[self.exlude:] = torch.where(correction_image[:ch-2] == 0., correction_image[:ch-2], image[self.exlude:])
        del correction_image

        return {'image': image, 'label': label}


class CorrCorrect(object):
    """Copied from Pytorch Documentation

    Correlation Correction
    """

    def __init__(self, summary_included=True, corr_form=c.corr['corr_form'], device=c.cuda['device'],
                 dtype=c.data['dtype']):
        self.corr_form = corr_form
        self.device = device
        self.dtype = dtype
        self.summary = summary_included

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # deleting information about not available offset pixels, need 2 dimensions otherwise correlation always 0
        correction_image = corr.get_corr(
            torch.rand(2, image.size(1), image.size(2), device=self.device, dtype=self.dtype),
            self.corr_form, device=self.device, dtype=self.dtype)
        if self.summary:
            image[2:] = torch.where(correction_image == 0., correction_image, image[2:])
        else:
            image = torch.where(correction_image == 0., correction_image, image)

        return {'image': image, 'label': label}


def preprocess_corr(corr_path, nb_corr_to_preserve=0, use_denoiser=False, test=False):
    """
    Method to preprocess correlation input data
    :param corr_path:
    :param nb_corr_to_preserve:
    :return:
    """
    corr_path = str(corr_path) + '/'
    files = sorted(glob(corr_path + '*.hkl'))
    imgs = torch.tensor(
        [load_numpy_from_h5py(file_name=f) for f in files if 'labels' not in f and '16' not in f])
    dims = imgs.size()[2:]  # 512 x 512

    for i in range(imgs.size(0)):

        ret = np.empty((nb_corr_to_preserve, dims[0], dims[1]))
        print('transforming ' + str(i) + 'th correlation image')
        corr_mean = torch.mean(imgs[i][nb_corr_to_preserve:], dim=0)

        for j in range(nb_corr_to_preserve):

            corrected_img = torch.mean(imgs[j], dim=0) - corr_mean
            corrected_img = torch.where(corrected_img < 0., torch.tensor(0., dtype=torch.double).cpu(), corrected_img)
            if use_denoiser:
                corrected_img = h.denoise(corrected_img.cpu().numpy(), weight=0.05, eps=0.00001)
            else:
                corrected_img = corrected_img.cpu().numpy()
            ret[j] = normalize_summary_img(torch.tensor(corrected_img)).cpu().numpy()

        save_numpy_to_h5py(data_array=ret, file_name='corr_nf_' + str(i),
                               file_path=str(corr_path) + 'transformed_' + str(nb_corr_to_preserve) + '/')

        if not test:
            labels = torch.tensor(
                [load_numpy_from_h5py(file_name=f) for f in files if 'labels' in f and '16' not in f])
            save_numpy_to_h5py(data_array=labels[i].detach().cpu().numpy(), file_name='corr_nf_' + str(i) + '_labels',
                                   file_path=str(corr_path) + 'transformed_' + str(nb_corr_to_preserve) + '/')

    pass


def create_corr_data(neurofinder_path, corr_form='small_star', slicing=c.corr['use_slicing'], slice_size=1,
                     test=False, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method that creates the corresponding correlation data from the neurofinder videos and returns them
    :param neurofinder_path:
    :param corr_form:
    :param slicing:
    :param slice_size:
    :return: Tensor: Number of Correlations (Depending on Corr_Form) x NbPixelsX x NbPixelsY
    """

    files = sorted(glob(neurofinder_path + '/images/*.tiff'))
    imgs = torch.tensor(array([imread(f) for f in files]).astype(np.float64), dtype=dtype,
                        device=device)
    dims = imgs.size()[1:]  # 512 x 512
    len = imgs.size(0)  # 3024

    assert (not slicing) or slice_size < len, 'Slicing Size must be smaller than the length of the Video'

    different_labels = c.data['different_labels']

    # if not using slicing correlations:
    if not slicing:
        corr_tensor = corr.get_corr(imgs, corr_form=corr_form, device=device, dtype=dtype)
    else:
        corr_tensor = corr.get_sliced_corr(imgs, corr_form=corr_form, slice_size=slice_size, device=device, dtype=dtype)

    if not test:
        # load the regions (training data only)
        with open(neurofinder_path + '/regions/regions.json') as f:
            regions = json.load(f)

        mask = array([tomask(s['coordinates'], dims) for s in regions])
        counter = 0

        if different_labels:
            for s in mask:
                mask[counter, :, :] = np.where(s == 1., 1. + counter, 0.)
                counter = counter + 1

        mask = torch.tensor(np.amax(mask, axis=0))

        corr_sample = {'correlations': corr_tensor, 'labels': mask}
    else:
        corr_sample = {'correlations': corr_tensor}

    return corr_sample


def get_corr_data(nf_folder, corr_path, file_name='corr_nf_', slicing=True, slice_size=100, corr_form='starmy',
                  test=False, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method that creates correlation and saves it to specified folder
    :param nf_folder:
    :param corr_path:
    :param file_name:
    :param slicing:
    :param corr_form:
    :param dtype:
    :param device:
    :return:
    """
    for index, folder in enumerate(sorted(os.listdir(nf_folder))):
        print(folder)
        corr_ = create_corr_data(neurofinder_path=str(nf_folder) + '/' + str(folder), slicing=slicing,
                                 slice_size=slice_size,
                                 corr_form=corr_form, test=test, dtype=dtype, device=device)
        if not test:
            save_numpy_to_h5py(data_array=corr_['correlations'].numpy(), label_array=corr_['labels'].numpy(),
                               file_name=file_name + str(index),
                               file_path=corr_path + '/')
        else:
            save_numpy_to_h5py(data_array=corr_['correlations'].numpy(),
                               file_name=file_name + str(index),
                               file_path=corr_path + '/')
    pass


def get_mean_img(video):
    """
    Method that calculates the summary image of nf datasets with respect to the mean
    :param video: tensor T x W x H
    :return: summary image of the neurofinder video W x H
    """
    return torch.mean(video, dim=0)


def get_var_img(video):
    """
    Method that calculates the summary image of nf datasets with respect to the variance
    :param video: tensor T x W x H
    :return: summary image of the neurofinder video W x H
    """
    return torch.var(video, dim=0)


def normalize_summary_img(summary_img, device=c.cuda['device'], dtype=c.data['dtype']):
    """
    Method that normalizes the mean summary image
    :param summary_img: tensor W x H
    :param device:
    :param dtype:
    :return: Normalized Mean summary image W x H
    """
    dims = summary_img.size()
    v = summary_img.view(-1)
    v_ = torch.mean(v, dim=0)
    v_v_ = v - v_
    v_v_n = torch.sqrt(torch.sum(v_v_ ** 2, dim=0))

    return (v_v_ / v_v_n).view(dims[0], dims[1])


# a = create_corr_data('data/neurofinder.00.00')
# print(a['correlations'].size(), a['labels'].size())


def create_summary_img(nf_folder, sum_folder='data/sum_img', test=False, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method that creates summary image of specified neurofinder folder
    :param nf_folder:
    :param dtype:
    :param device:
    :param test:
    :return:
    """

    files = sorted(glob(nf_folder + '/images/*.tiff'))

    imgs = torch.tensor(array([imread(f) for f in files]).astype(np.float64), dtype=dtype,
                        device=device)

    mean_summar = get_mean_img(imgs)
    h = imgs - mean_summar
    h_ = get_mean_img(torch.where(h < 0., torch.tensor(0.), h))
    mean_summar = normalize_summary_img(h_)

    var_summar = get_var_img(imgs)
    g = var_summar
    g_ = get_var_img(torch.where(g < 0., torch.tensor(0.), g))
    g_ = torch.sqrt(g)
    var_summar = normalize_summary_img(g_)

    con = 0
    if test:
        con = 5
    save_numpy_to_h5py(data_array=mean_summar.detach().cpu().numpy(), file_name=str(nf_folder)[-(5+con):-con] + '_mean',
                       file_path=str(sum_folder) + '/')
    save_numpy_to_h5py(data_array=var_summar.detach().cpu().numpy(), file_name=str(nf_folder)[-(5+con):-con] + '_var',
                       file_path=str(sum_folder) + '/')
    pass


def get_summary_img(nf_folder, sum_folder='data/sum_img', test=False, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method that creates summmary images of the Neurofinder Datasets
    :param nf_folder:
    :param dtype:
    :param device:
    :return:
    """
    dic = os.listdir(nf_folder)
    print(dic)
    for x in dic:
        print('Summarizing ' + str(x) + ' ...')
        create_summary_img(str(nf_folder) + '/' + str(x), sum_folder=sum_folder, test=test, dtype=dtype, device=device)
    pass


def generate_data(nf_folder, corr_path, slicing, slice_size, sum_folder, nb_corr_to_preserve, generate_summary=False,
                  use_denoiser=False,
                  testset=False):
    """
    Method to generate input data give the following settings
    :param nf_folder: Folder which  has the Neurofinder Data
    :param corr_path: Folder where correlations should be stored
    :param slicing: Whether to use slicing
    :param slice_size: specify slice size
    :param sum_folder: folder where summary images should be stored
    :param nb_corr_to_preserve: number or correlations dimensions to preserve in preprocess correlations
    :param generate_summary: whether to generate summary images
    :param use_denoiser: whether to use denoiser (should be False)
    :param testset: whether nf_folder contains test set or training set
    :return: nothing
    """
    dtype = torch.double
    cpu = torch.device('cpu')

    get_corr_data(nf_folder=nf_folder, corr_path=corr_path,
                  slicing=slicing, test=testset,
                  slice_size=slice_size, device=cpu, dtype=dtype)
    if generate_summary:
        get_summary_img(nf_folder=nf_folder, sum_folder=sum_folder,
                        test=testset, device=cpu, dtype=dtype)
    preprocess_corr(corr_path=corr_path, nb_corr_to_preserve=nb_corr_to_preserve,
                    use_denoiser=use_denoiser)
    pass


def save_numpy_to_h5py(data_array, file_name, file_path, label_array=None, use_compression=c.data['use_compression']):
    """
    Method to save correlation data and label numpy arrays to disk either compressed or not
    :param use_compression: whether to use compression or not
    :param label_array: array of labels for data
    :param data_array: Numpy Array of data to be saved
    :param file_name: Name of The File that should be created
    :param file_path: Path to the File that should be created
    :return:
    """
    if not use_compression:
        # Dump data to file
        hkl.dump(data_array, str(file_path) + str(file_name) + '.hkl', mode='w')
        print('File ' + str(file_path) + str(file_name) + '.hkl' +
              ' saved uncompressed: %i bytes' % os.path.getsize(str(file_path) + str(file_name) + '.hkl'))
        if not label_array is None:
            hkl.dump(label_array, str(file_path) + str(file_name) + '_labels.hkl', mode='w')
            # Dump Labels to file
            print('File ' + str(file_path) + str(file_name) + '_labels.hkl' +
                  ' saved uncompressed: %i bytes' % os.path.getsize(str(file_path) + str(file_name) + '_labels.hkl'))
    else:
        # Dump data, with compression to file
        hkl.dump(data_array, str(file_path) + str(file_name) + '_gzip.hkl', mode='w', compression='gzip')
        print('File ' + str(file_path) + str(file_name) + '_gzip.hkl' +
              ' saved compressed:   %i bytes' % os.path.getsize(str(file_path) + str(file_name) + '_gzip.hkl'))
        if not label_array is None:
            # Dump labels, with compression to file
            hkl.dump(label_array, str(file_path) + str(file_name) + '_labels_gzip.hkl', mode='w', compression='gzip')
            print('File ' + str(file_path) + str(file_name) + '_labels_gzip.hkl' +
                  ' saved compressed:   %i bytes' % os.path.getsize(
                str(file_path) + str(file_name) + '_labels_gzip.hkl'))
    pass


def load_numpy_from_h5py(file_name):
    """
    Method that loads correlation data and labels numpy arrays from h5py file
    :param use_compression: whether data to be loaded is compressed or not
    :param file_name:
    :param file_path: if /.../dir/
    :return: numpy array data and numpy array labels loaded from file
    """
    return hkl.load(str(file_name))

# neurofinder_dataset = NeurofinderDataset('data/neurofinder.00.00')
#
# a = create_corr_data(neurofinder_dataset=neurofinder_dataset, corr_form='small_star', slicing=False, slice_size=1000)
# print(a['correlations'].size())
# print(a['labels'].size())
