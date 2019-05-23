from numpy import array, zeros
import numpy as np
from scipy.misc import imread
from glob import glob
import json
import config as c
import torch


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


def create_corr_data_outdated(neurofinder_dataset, corr_form='small_star', slicing=c.corr['use_slicing'], slice_size=1):
    """
    Method that creates the corresponding correlation data from the neurofinder videos and returns them
    :param neurofinder_dataset:
    :param corr_form:
    :param slicing:
    :param slice_size:
    :return: Tensor: Number of Correlations (Depending on Corr_Form) x NbPixelsX x NbPixelsY
    """

    length = neurofinder_dataset.__len__()

    assert (not slicing) or slice_size < length, 'Slicing Size must be smaller than the length of the Video'

    data_tensor = torch.from_numpy(neurofinder_dataset[0]['image'].astype(float)).unsqueeze(dim=0)
    target_tensor = torch.from_numpy(neurofinder_dataset[0]['label'].astype(float)).unsqueeze(dim=0)
    for i in range(1, length):
        '''
        TOOOOOOOO SLOOOWWWWWW
        '''
        print(i) # helps for seeing it work
        data_tensor = torch.cat((data_tensor, torch.from_numpy(neurofinder_dataset[i]['image'].astype(float)).unsqueeze(dim=0)), dim=0)

    # if not using slicing correlations:
    if not slicing:
        corr_tensor = corr.get_corr(data_tensor, corr_form=corr_form)
    else:
        corr_tensor = corr.get_sliced_corr(data_tensor, corr_form=corr_form, slice_size=slice_size)

    corr_sample = {'correlations': corr_tensor, 'labels': target_tensor}

    return corr_sample
