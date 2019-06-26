import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
import umap
# %matplotlib inline
import matplotlib.pyplot as plt
import torch


def plot_sk_nn(data, labels):
    data = data.cpu().numpy()
    for i in range(data.shape[1] - 1):
        # Create plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(x=data[:, i], y=data[:, i + 1], alpha=0.8, c=labels[:, 0], edgecolors='none', s=30)

        plt.title('Clustered Embeddings')
        plt.legend(loc=2)
        plt.show()


def plot_kmean(data, clusters_index, cluster_means):
    data = data.cpu().numpy()

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(x=data[:, 0], y=data[:, 1], alpha=0.8, c=clusters_index, edgecolors='none', s=30)
    ax.scatter(x=cluster_means[:, 0], y=cluster_means[:, 1], alpha=0.8, c='black', edgecolors='none', s=30)

    plt.title('Clustered Embeddings')
    plt.legend(loc=2)
    plt.show()

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


def draw_umap(n_neighbors=10, min_dist=0.1, n_components=2, metric='cosine', title='Embeddings', data=None, color=None):

    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )

    v = data.cpu().numpy()
    v_ = np.mean(v, axis=0)
    v_v_ = v - v_
    v_v_n = np.std(v, axis=0)
    v = v_v_ / v_v_n
    v = v.T

    lab = color.cpu().numpy()

    u = fit.fit_transform(v)
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=lab)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=lab)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=lab, s=100)
    plt.title(title, fontsize=18)
    return fig
