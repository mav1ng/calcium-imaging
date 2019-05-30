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

# a = torch.randint(0, 10, (2, 2, 2)).float()
# a = a.view(2, -1)
# print(a)
# print(a.norm(dim=0))
# a_norm = a / a.norm(dim=0)[None, :]
# b_norm = a / a.norm(dim=0)[None, :]
# print(a_norm)
# print(b_norm)
# res = torch.mm(a_norm.transpose(0, 1), b_norm)
# print(torch.mm(res)
#
