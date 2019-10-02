import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
import umap
# %matplotlib inline
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
import pandas as pd
from PIL import Image
import numpy as np


def plot_pred_back(data, labels):
    """
        Visualizes The embeddings with dimesnsion reduction via PCA
        :param data: tensor C x w x h
        :param labels: tensor w x h
        """

    (c, w, h) = data.size()

    l = labels.squeeze(0).cpu().numpy()

    for i in range(c):
        d = data[i].cpu().numpy()

        f, axarr = plt.subplots(2)
        axarr[0].imshow(d)
        axarr[1].imshow(l)

        plt.title('Predicted Background (upper) vs Ground Truth (lower)')

        plt.show()


def plot_input(data, labels):
    """
        Visualizes The embeddings with dimesnsion reduction via PCA
        :param data: tensor C x w x h
        :param labels: tensor w x h
        """

    (c, w, h) = data.size()

    l = labels.squeeze(0).cpu().numpy()

    for i in range(c):
        d = data[i].cpu().numpy()

        f, axarr = plt.subplots(2)
        axarr[0].imshow(d)
        axarr[1].imshow(l)

        plt.title('Actual Mean (upper) vs Ground Truth (lower)')

        plt.show()



def plot_emb_pca(data, labels):
    """
    Visualizes The embeddings with dimesnsion reduction via PCA
    :param data: tensor C x w x h
    :param labels: tensor w x h
    """

    pca = PCA(n_components=3)
    l = labels

    try:
        (c, w, h) = data.size()

        d_ = torch.mean(data, dim=0)
        d_n = data - d_
        d_n_ = torch.sqrt(torch.sum(d_n ** 2, dim=0))
        d = d_n / d_n_

        d = d.view(c, -1).t().cpu().numpy()
    except (ValueError, TypeError):
        (c, w, h) = data.shape

        d_ = np.mean(data, axis=0)
        d_n = data - d_
        d_n_ = np.sqrt(np.sum(d_n ** 2, axis=0))
        d = d_n / d_n_
        d = np.where(np.isnan(d) != 1, d, 0.)
        d = d.reshape(c, -1).T

    principalComponents = pca.fit_transform(d)

    pDf = principalComponents.reshape(w, h, 3)

    print('PCA max and min', np.max(pDf), np.min(pDf))

    f, axarr = plt.subplots(2)
    axarr[0].imshow(pDf)
    axarr[1].imshow(l)

    plt.title('PCA Actual (upper) vs Ground Truth (lower)')

    plt.show()

    try:
        data_ = data.view(3, -1).t().cpu().numpy()
        plt.scatter(data_[:, 0], data_[:, 1], data_[:, 2], cmap='tab20b')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data_[:, 0], data_[:, 1], data_[:, 2], cmap='tab20b')
        plt.show()
    except TypeError:
        pDf = pDf.reshape(-1, 3)
        plt.scatter(pDf[:, 0], pDf[:, 1], pDf[:, 2])
        plt.show()


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


def plot_sk_img(labels, ground_truth):
    """
    :param labels: N x 1, N number of pixels
    :return: predicted labels on image
    """
    l = ground_truth.squeeze(0).cpu().numpy()
    n = labels.shape[0]
    f, axarr = plt.subplots(2)
    axarr[0].imshow(labels.reshape(int(np.sqrt(n)), int(np.sqrt(n))))
    axarr[1].imshow(l)

    plt.title('Predicted Embeddings (upper) vs Ground Truth (lower)')
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


def draw_umap(n_neighbors=10, min_dist=0.1, n_components=2, metric='euclidean', title='Embeddings', data=None, color=None):

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
