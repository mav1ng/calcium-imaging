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

import config


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
        self.different_labels = config.data['different_labels']

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

    if config.data['different_labels']:
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


neurofinder_dataset = NeurofinderDataset('data/neurofinder.00.00')

a = create_corr_data(neurofinder_dataset=neurofinder_dataset, corr_form='small_star', slicing=False, slice_size=1000)
print(a['correlations'].size())
print(a['labels'].size())
