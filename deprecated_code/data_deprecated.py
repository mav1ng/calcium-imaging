from numpy import array, zeros
import numpy as np
from scipy.misc import imread
from glob import glob
import json
import config as c

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
