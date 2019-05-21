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
import torch.tensor

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
        self.dims = self.imgs.shape[1:]
        self.len = self.imgs.shape[0]
        self.transform = transform
        self.different_labels = config.data['different_labels']

        # load the regions (training data only)
        with open(neurofinder_path + '/regions/regions.json') as f:
            self.regions = json.load(f)

        self.masks = array([tomask(s['coordinates'], self.dims) for s in self.regions])
        self.counter = 0

        if self.different_labels:
            for s in self.masks:
                self.masks[self.counter, :, :] = np.where(s == 1., 1. + self.counter, 0.)
                self.counter = self.counter + 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {'image': self.imgs[idx, :, :], 'label': self.masks[idx, :, :]}
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


def get_corr_data(neurofinder_dataset, corrform):
    length = neurofinder_dataset.__len__()
    data_tensor = torch.tensor.as_tensor()(neurofinder_dataset[0])
    for i in range(length):
        pass
    return

def get_sliced_corr_data(neurofinder_dataset, corrform, slice_size):
    return


neurofinder_dataset = NeurofinderDataset('data/neurofinder.00.00')
sample = neurofinder_dataset[0]['image']
data_tensor = torch.Tensor()
x = torch.Tensor
print(x)
print(sample)
plt.imshow(sample)
