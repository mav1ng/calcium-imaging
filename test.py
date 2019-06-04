import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import h5py
import json
import matplotlib.pyplot as plt
from numpy import array, zeros
import numpy as np
from scipy.misc import imread
from glob import glob
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import config as c
import network as n
import data
import corr

import json
import socket
import urllib.request as req

dtype = c.data['dtype']
device = c.cuda['device']

torch.cuda.empty_cache()

test = torch.rand(5, 10, 64, 64, dtype=dtype, device=device)
model = n.UNetMS()
model.to(device)
model.type(dtype)


