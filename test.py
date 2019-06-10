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
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import config as c
import network as n
import data
import corr
import training as t

a = torch.rand((10, 1, 2, 2, 2), device=torch.device('cuda'))
b = torch.randint(0, 10, (10, 2, 2), device=torch.device('cuda'), dtype=torch.float)

print(n.embedding_loss(a[0, 0], b[0]))
print(n.embedding_loss_new(a[0, 0], b[0]))

print(n.get_batch_embedding_loss(a, b))
print(n.get_batch_embedding_loss_new(a, b))