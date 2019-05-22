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


def tomask(coords, dims):
    mask = zeros(dims)
    mask[list(zip(*coords))] = 1.
    return mask


class NeurofinderDataset(Dataset):
    """Neurofinder Dataset"""

    def __init__(self, neurofinder_path, transform=None):
        """
        :param neurofinder_path: Path to the neurofinder dataset
        :param transform:
        """

        self.neurofinder_path = neurofinder_path
        self.files = sorted(glob(neurofinder_path + '/images/*.tiff'))
        self.imgs = array([imread(f) for f in self.files])
        self.dims = self.imgs.shape[1:]         # 512 x 512
        self.len = self.imgs.shape[0]           # 3024
        self.transform = transform
        self.different_labels = c.data['different_labels']

        # load the regions (training data only)
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
        sample = {'image': self.imgs[idx, :, :], 'label': self.mask}
        if self.transform:
            sample = self.transform(sample)
        return sample


def load_data(neurofinder_path):
    # load the images
    files = sorted(glob(neurofinder_path + '/images/*.tiff'))
    imgs = array([imread(f) for f in files])
    dims = imgs.shape[1:]

    # load the regions (training data only)
    with open(neurofinder_path + '/regions/regions.json') as f:
        regions = json.load(f)

    for s in regions:
        print(s['coordinates'])

    def tomask(coords):
        mask = zeros(dims)
        mask[list(zip(*coords))] = 1.
        return mask

    masks = array([tomask(s['coordinates']) for s in regions])
    counter = 0

    if c.data['different_labels']:
        for s in masks:
            masks[counter, :, :] = np.where(s == 1., 1. + counter, 0.)
            counter = counter + 1.


def create_corr_data(neurofinder_dataset, corr_form='small_star', slicing=False, slice_size=1):
    """
    Method that creates the corresponding correlation data from the neurofinder videos and returns them
    :param neurofinder_dataset:
    :param corr_form:
    :param slicing:
    :param slice_size:
    :return: Number of Correlations (Depending on Corr_Form) x NbPixelsX x NbPixelsY
    """

    length = neurofinder_dataset.__len__()
    # length = 10        # just for testing purposes to speed up testing

    assert (not slicing) or slice_size < length, 'Slicing Size must be smaller than the length of the Video'

    data_tensor = torch.from_numpy(neurofinder_dataset[0]['image'].astype(float)).unsqueeze(dim=0)
    target_tensor = torch.from_numpy(neurofinder_dataset[0]['label'].astype(float)).unsqueeze(dim=0)
    for i in range(1, length):
        data_tensor = torch.cat((data_tensor, torch.from_numpy(neurofinder_dataset[i]['image'].astype(float)).unsqueeze(dim=0)), dim=0)

    # if not using slicing correlations:
    if not slicing:
        corr_tensor = corr.get_corr(data_tensor, corr_form=corr_form)
    else:
        corr_tensor = corr.get_sliced_corr(data_tensor, corr_form=corr_form, slice_size=slice_size)

    corr_sample = {'correlations': corr_tensor, 'labels': target_tensor}

    return corr_sample


def save_numpy_to_h5py(array_object, file_name, file_path, use_compression=c.data['use_compression']):
    """
    Method to save numpy arrays to disk either compressed or not
    :param array_object: Numpy Array to be saved
    :param file_name: Name of The File that should be created
    :param file_path: Path to the File that should be created
    :param compression: Boolean Whether to use compression or not
    :return:
    """
    if not use_compression:
        # Dump to file
        hkl.dump(array_object, str(file_path) + str(file_name) + '.hkl', mode='w')
        print('File ' + str(file_path) + str(file_name) + '.hkl' +
              ' saved uncompressed: %i bytes' % os.path.getsize(str(file_path) + str(file_name) + '.hkl'))
    else:
        # Dump data, with compression
        hkl.dump(array_object, str(file_path) + str(file_name) + '_gzip.hkl', mode='w', compression='gzip')
        print('File ' + str(file_path) + str(file_name) + '_gzip.hkl' +
              ' saved compressed:   %i bytes' % os.path.getsize(str(file_path) + str(file_name) + '_gzip.hkl'))
    pass


def load_numpy_from_h5py(file_name, file_path):
    """
    Method that loads numpy array from h5py file
    :param file_name: is 'name_ifcompressed.hkl'
    :param file_path: if /.../dir/
    :return: numpy array loaded from file
    """
    return hkl.load(str(file_path) + str(file_name))


# neurofinder_dataset = NeurofinderDataset('data/neurofinder.00.00')
#
# a = create_corr_data(neurofinder_dataset=neurofinder_dataset, corr_form='small_star', slicing=False, slice_size=1000)
# print(a['correlations'].size())
# print(a['labels'].size())


