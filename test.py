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
import network as n
import data


import json
import socket
import urllib.request as req


dtype = c.data['dtype']
device = c.cuda['device']

torch.cuda.empty_cache()

input_test = torch.rand(1, 10, 32, 32, dtype=dtype, device=device, requires_grad=True)
labels = torch.randint(0, 10, (32, 32), dtype=dtype, device=device)

model = n.UNet()
model.to(device, dtype)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for t in range(100):
    # Forward pass: Compute predicted y by passing x to the model
    embedding_list = model(input_test)

    print(embedding_list.size())

    # Compute and print loss
    loss = n.embedding_loss(embedding_list.view(10, 32, 32), labels, device=device, dtype=dtype)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
