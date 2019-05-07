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

import config


a = torch.randn(1, 1, 32, 32)
b = torch.randn(32, 32)

filename = 'data/data_corrs/corrs_pearson.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = f['neurofinder.00.00']

print(data)


# load the images
files = sorted(glob('data/neurofinder.00.00/images/*.tiff'))
imgs = array([imread(f) for f in files])
dims = imgs.shape[1:]

# load the regions (training data only)
with open('data/neurofinder.00.00/regions/regions.json') as f:
    regions = json.load(f)

for s in regions:
    print(s['coordinates'])


def tomask(coords):
    mask = zeros(dims)
    mask[list(zip(*coords))] = 1
    return mask


masks = array([tomask(s['coordinates']) for s in regions])
counter = 0
print(masks.shape)
for s in masks:
    if not config.data['different_labels']:
        counter = 0
    masks[counter, :, :] = np.where(s == 1, 1 + counter, s)
    counter = counter + 1
    print(counter)


print(np.argmax(np.argmax(masks, axis=0), axis=0))


# show the outputs
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(imgs.sum(axis=0), cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(masks.sum(axis=0), cmap='gray')
plt.show()
