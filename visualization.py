import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
import umap
# %matplotlib inline
import matplotlib.pyplot as plt
import torch


def plot3Dembeddings(embeddings):
    xlist = []
    ylist = []
    zlist = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(embeddings.size(-1)):
        for k in range(embeddings.size(-2)):
            axis = embeddings.detach().to('cpu').numpy()[0, -1, :, i, k]
            xlist.append(axis[0])
            ylist.append(axis[1])
            zlist.append(axis[2])
    ax.scatter(xs=xlist, ys=ylist, zs=zlist, zdir='z', s=20)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    plt.show()


def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title='', data=None, color=None):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=color)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=color)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=color, s=100)
    plt.title(title, fontsize=18)
