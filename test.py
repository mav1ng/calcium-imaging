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

import config as c
import data


import json
import socket
import urllib.request as req

a = torch.rand(3, 3, 3)
print(a[0, :, :])
a = torch.roll(a.view(3, -1), shifts=1, dims=1).view(3, 3, 3)
print(a[0, :, :])