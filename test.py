import torch
import corr as c
import numpy as np
import matplotlib.pyplot as plt
import clustering as cl
import visualization as v

A = np.concatenate([np.random.randn(1000, 2), np.random.randn(1000, 2)+3, np.random.randn(1000, 2)+6], axis=0)

ind, mean = cl.cluster_kmean(data=A)
v.plot_kmean(A, ind, mean)