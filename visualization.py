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
import data
import helpers as h
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


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

def plot_learning_curve(model_name, figsize, cutoff_1=0, cutoff_2=0, azrael=False):

    d_val_emb_long = np.array(data.read_from_json('data/learning_curves/' + str(model_name) + '_long_val_emb.json'))
    d_emb_long = np.array(data.read_from_json('data/learning_curves/' + str(model_name) + '_long_emb.json'))
    d_val_emb_pre = np.array(data.read_from_json('data/learning_curves/pre_' + str(model_name) + '_val_emb.json'))
    d_emb_pre = np.array(data.read_from_json('data/learning_curves/pre_' + str(model_name) + '_emb.json'))

    d_val_emb_long = np.where(np.isnan(d_val_emb_long) == 1, 0., d_val_emb_long)
    d_emb_long = np.where(np.isnan(d_emb_long) == 1, 0., d_emb_long)
    d_val_emb_pre = np.where(np.isnan(d_val_emb_pre) == 1, 0., d_val_emb_pre)
    d_emb_pre = np.where(np.isnan(d_emb_pre) == 1, 0., d_emb_pre)

    val_emb_long = d_val_emb_long[:, 2]
    emb_long = d_emb_long[:, 2]
    val_emb_pre = d_val_emb_pre[:, 2]
    emb_pre = d_emb_pre[:, 2]

    norm_val_emb_long = np.linalg.norm(val_emb_long)
    norm_emb_long = np.linalg.norm(emb_long)
    norm_val_emb_pre = np.linalg.norm(val_emb_pre)
    norm_emb_pre = np.linalg.norm(emb_pre)

    val_emb_long = val_emb_long / norm_val_emb_long
    emb_long = emb_long / norm_emb_long
    val_emb_pre = val_emb_pre / norm_val_emb_pre
    emb_pre = emb_pre / norm_emb_pre

    if not azrael:
        d_val_cel_long = np.array(data.read_from_json('data/learning_curves/' + str(model_name) + '_long_val_cel.json'))
        d_cel_long = np.array(data.read_from_json('data/learning_curves/' + str(model_name) + '_long_cel.json'))
        d_val_cel_pre = np.array(data.read_from_json('data/learning_curves/pre_' + str(model_name) + '_val_cel.json'))
        d_cel_pre = np.array(data.read_from_json('data/learning_curves/pre_' + str(model_name) + '_cel.json'))

        d_val_cel_long = np.where(np.isnan(d_val_cel_long) == 1, 0., d_val_cel_long)
        d_cel_long = np.where(np.isnan(d_cel_long) == 1, 0., d_cel_long)
        d_val_cel_pre = np.where(np.isnan(d_val_cel_pre) == 1, 0., d_val_cel_pre)
        d_cel_pre = np.where(np.isnan(d_cel_pre) == 1, 0., d_cel_pre)

        time_val_long = (d_val_cel_long[:, 0] - d_val_cel_long[0, 0]) / 3600
        time_long = (d_cel_long[:, 0] - d_cel_long[0, 0]) / 3600
        time_val_pre = (d_val_cel_pre[:, 0] - d_val_cel_pre[0, 0]) / 3600
        time_pre = (d_cel_pre[:, 0] - d_cel_pre[0, 0]) / 3600

        val_cel_long = d_val_cel_long[:, 2]
        cel_long = d_cel_long[:, 2]
        val_cel_pre = d_val_cel_pre[:, 2]
        cel_pre = d_cel_pre[:, 2]

        norm_val_cel_long = np.linalg.norm(val_cel_long)
        norm_cel_long = np.linalg.norm(cel_long)
        norm_val_cel_pre = np.linalg.norm(val_cel_pre)
        norm_cel_pre = np.linalg.norm(cel_pre)

        val_cel_long = val_cel_long / norm_val_cel_long
        cel_long = cel_long / norm_cel_long
        val_cel_pre = val_cel_pre / norm_val_cel_pre
        cel_pre = cel_pre / norm_cel_pre

        comb_val_long = (val_cel_long + val_emb_long) / 2
        comb_val_pre = (val_cel_pre + val_emb_pre) / 2
        comb_long = (cel_long + emb_long) / 2
        comb_pre = (cel_pre + emb_pre) / 2

        comb_val_long = h.zero_to_one(comb_val_long)
        comb_val_pre = h.zero_to_one(comb_val_pre)
        comb_long = h.zero_to_one(comb_long)
        comb_pre = h.zero_to_one(comb_pre)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        cmap = plt.cm.get_cmap('tab20b')

        time_val_long, comb_val_long = zip(
            *sorted(zip(time_val_long, comb_val_long), key=lambda time_val_long: time_val_long[0]))
        time_long, comb_long = zip(
            *sorted(zip(time_long, comb_long), key=lambda time_long: time_long[0]))
        time_val_pre, comb_val_pre = zip(
            *sorted(zip(time_val_pre, comb_val_pre), key=lambda time_val_pre: time_val_pre[0]))
        time_pre, comb_pre = zip(
            *sorted(zip(time_pre, comb_pre), key=lambda time_pre: time_pre[0]))

        plt.plot(time_val_long, comb_val_long, MarkerSize=3, color=cmap(0.), marker='o', alpha=0.4,
                 label='Random Initialization',
                 linewidth=2.,
                 linestyle='-')

        plt.plot(time_val_pre, comb_val_pre, MarkerSize=3, color=cmap(14 / 20), marker='o', alpha=0.4,
                 label='Pretrained Model', linewidth=2.,
                 linestyle='-')

        plt.title(str(model_name).upper() + ': Validation Loss vs. Time [h]')
        plt.xlabel('Time [h]')
        plt.ylabel('Model Validation Loss Offset to [0, 1]')

        if cutoff_1 != 0:
            plt.xlim(0, cutoff_1)
        ax.legend()
        plt.tight_layout()
        plt.savefig('x_images/plots/lc_val_' + str(model_name) + '.pdf')
        plt.show()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        plt.plot(time_long, comb_long, MarkerSize=3, color=cmap(16 / 20), marker='o', alpha=0.4,
                 label='Random Initialization',
                 linewidth=2.,
                 linestyle='-')

        plt.plot(time_pre, comb_pre, MarkerSize=3, color=cmap(3 / 20), marker='o', alpha=0.4,
                 label='Pretrained Model', linewidth=2.,
                 linestyle='-')

        plt.title(str(model_name).upper() + ': Training Loss vs. Time [h]')
        plt.xlabel('Time [h]')
        plt.ylabel('Model Training Loss Offset to [0, 1]')

        if cutoff_2 != 0:
            plt.xlim(0, cutoff_2)
        ax.legend()
        plt.tight_layout()
        plt.savefig('x_images/plots/lc_' + str(model_name) + '.pdf')
        plt.show()

    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        cmap = plt.cm.get_cmap('tab20b')

        time_val_long = (d_val_emb_long[:, 0] - d_val_emb_long[0, 0]) / 3600
        time_long = (d_emb_long[:, 0] - d_emb_long[0, 0]) / 3600
        time_val_pre = (d_val_emb_pre[:, 0] - d_val_emb_pre[0, 0]) / 3600
        time_pre = (d_emb_pre[:, 0] - d_emb_pre[0, 0]) / 3600

        time_val_long, val_emb_long = zip(
            *sorted(zip(time_val_long, val_emb_long), key=lambda time_val_long: time_val_long[0]))
        time_long, emb_long = zip(
            *sorted(zip(time_long, emb_long), key=lambda time_long: time_long[0]))
        time_val_pre, val_emb_pre = zip(
            *sorted(zip(time_val_pre, val_emb_pre), key=lambda time_val_pre: time_val_pre[0]))
        time_pre, emb_pre = zip(
            *sorted(zip(time_pre, emb_pre), key=lambda time_pre: time_pre[0]))

        plt.plot(time_val_long, val_emb_long, MarkerSize=3, color=cmap(0.), marker='o', alpha=0.4,
                 label='Random Initialization',
                 linewidth=2.,
                 linestyle='-')

        plt.plot(time_val_pre, val_emb_pre, MarkerSize=3, color=cmap(14 / 20), marker='o', alpha=0.4,
                 label='Pretrained Model', linewidth=2.,
                 linestyle='-')

        plt.title(str(model_name).upper() + ': Validation Loss vs. Time [h]')
        plt.xlabel('Time [h]')
        plt.ylabel('Model Validation Loss Offset to [0, 1]')

        if cutoff_1 != 0:
            plt.xlim(0, cutoff_1)
        ax.legend()
        plt.tight_layout()
        plt.savefig('x_images/plots/lc_val_' + str(model_name) + '.pdf')
        plt.show()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        plt.plot(time_long, emb_long, MarkerSize=3, color=cmap(16 / 20), marker='o', alpha=0.4,
                 label='Random Initialization',
                 linewidth=2.,
                 linestyle='-')

        plt.plot(time_pre, emb_pre, MarkerSize=3, color=cmap(3 / 20), marker='o', alpha=0.4,
                 label='Pretrained Model', linewidth=2.,
                 linestyle='-')

        plt.title(str(model_name).upper() + ': Training Loss vs. Time [h]')
        plt.xlabel('Time [h]')
        plt.ylabel('Model Training Loss Offset to [0, 1]')

        if cutoff_2 != 0:
            plt.xlim(0, cutoff_2)
        ax.legend()
        plt.tight_layout()
        plt.savefig('x_images/plots/lc_' + str(model_name) + '.pdf')
        plt.show()


def plot_learning_curve_2(model_name, figsize, cutoff_1=0, cutoff_2=0, azrael=False):
    d_val_emb_long = np.array(data.read_from_json('data/learning_curves/' + str(model_name) + '_long_val_emb.json'))
    d_emb_long = np.array(data.read_from_json('data/learning_curves/' + str(model_name) + '_long_emb.json'))
    d_val_emb_pre = np.array(data.read_from_json('data/learning_curves/pre_' + str(model_name) + '_val_emb.json'))
    d_emb_pre = np.array(data.read_from_json('data/learning_curves/pre_' + str(model_name) + '_emb.json'))

    d_val_emb_long = np.where(np.isnan(d_val_emb_long) == 1, 0., d_val_emb_long)
    d_emb_long = np.where(np.isnan(d_emb_long) == 1, 0., d_emb_long)
    d_val_emb_pre = np.where(np.isnan(d_val_emb_pre) == 1, 0., d_val_emb_pre)
    d_emb_pre = np.where(np.isnan(d_emb_pre) == 1, 0., d_emb_pre)

    val_emb_long = d_val_emb_long[:, 2]
    emb_long = d_emb_long[:, 2]
    val_emb_pre = d_val_emb_pre[:, 2]
    emb_pre = d_emb_pre[:, 2]

    if not azrael:
        d_val_cel_long = np.array(data.read_from_json('data/learning_curves/' + str(model_name) + '_long_val_cel.json'))
        d_cel_long = np.array(data.read_from_json('data/learning_curves/' + str(model_name) + '_long_cel.json'))
        d_val_cel_pre = np.array(data.read_from_json('data/learning_curves/pre_' + str(model_name) + '_val_cel.json'))
        d_cel_pre = np.array(data.read_from_json('data/learning_curves/pre_' + str(model_name) + '_cel.json'))

        d_val_cel_long = np.where(np.isnan(d_val_cel_long) == 1, 0., d_val_cel_long)
        d_cel_long = np.where(np.isnan(d_cel_long) == 1, 0., d_cel_long)
        d_val_cel_pre = np.where(np.isnan(d_val_cel_pre) == 1, 0., d_val_cel_pre)
        d_cel_pre = np.where(np.isnan(d_cel_pre) == 1, 0., d_cel_pre)

        time_val_long = (d_val_cel_long[:, 0] - d_val_cel_long[0, 0]) / 3600
        time_long = (d_cel_long[:, 0] - d_cel_long[0, 0]) / 3600
        time_val_pre = (d_val_cel_pre[:, 0] - d_val_cel_pre[0, 0]) / 3600
        time_pre = (d_cel_pre[:, 0] - d_cel_pre[0, 0]) / 3600

        val_cel_long = d_val_cel_long[:, 2]
        cel_long = d_cel_long[:, 2]
        val_cel_pre = d_val_cel_pre[:, 2]
        cel_pre = d_cel_pre[:, 2]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        cmap = plt.cm.get_cmap('tab20b')

        time_val_long, val_emb_long = zip(
            *sorted(zip(time_val_long, val_emb_long), key=lambda time_val_long: time_val_long[0]))
        time_long, emb_long = zip(
            *sorted(zip(time_long, emb_long), key=lambda time_long: time_long[0]))
        time_val_pre, val_emb_pre = zip(
            *sorted(zip(time_val_pre, val_emb_pre), key=lambda time_val_pre: time_val_pre[0]))
        time_pre, emb_pre = zip(
            *sorted(zip(time_pre, emb_pre), key=lambda time_pre: time_pre[0]))
        time_val_long, val_cel_long = zip(
            *sorted(zip(time_val_long, val_cel_long), key=lambda time_val_long: time_val_long[0]))
        time_long, cel_long = zip(
            *sorted(zip(time_long, cel_long), key=lambda time_long: time_long[0]))
        time_val_pre, val_cel_pre = zip(
            *sorted(zip(time_val_pre, val_cel_pre), key=lambda time_val_pre: time_val_pre[0]))
        time_pre, cel_pre = zip(
            *sorted(zip(time_pre, cel_pre), key=lambda time_pre: time_pre[0]))

        plt.plot(time_val_long, val_emb_long, MarkerSize=6, color=cmap(14/20), marker='', alpha=1, markevery=5,
                 label='Embedding Loss: Random Initialization',
                 linewidth=2.,
                 linestyle=':')

        plt.plot(time_val_pre, val_emb_pre, MarkerSize=6, color=cmap(0.), marker='', alpha=1, markevery=5,
                 label='Embedding Loss: Pretrained Model', linewidth=2.,
                 linestyle=':')

        plt.plot(time_val_long, val_cel_long, MarkerSize=6, color=cmap(14 / 20), marker='', alpha=1, markevery=5,
                 label='Cross Entropy Loss: Random Initialization',
                 linewidth=2.,
                 linestyle='-')

        plt.plot(time_val_pre, val_cel_pre, MarkerSize=6, color=cmap(0.), marker='', alpha=1, markevery=5,
                 label='Cross Entropy Loss: Pretrained Model', linewidth=2.,
                 linestyle='-')

        plt.title(str(model_name).upper() + ': Validation Loss vs. Time [h]')
        plt.xlabel('Time [h]')
        plt.ylabel('Model Validation Loss')

        random_patch = mpatches.Patch(color=cmap(14 / 20), label='Random Initialization')
        pre_trained_patch = mpatches.Patch(color=cmap(0.), label='Pretrained Model')

        emb_line = mlines.Line2D([], [], color='black', marker='', linestyle=':',
                                  markersize=15, label='Embedding Loss')
        cel_line = mlines.Line2D([], [], color='black', marker='', linestyle='-',
                                  markersize=15, label='Cross Entropy Loss')

        ax.legend(handles=[random_patch, pre_trained_patch, emb_line, cel_line])

        if cutoff_1 != 0:
            plt.xlim(0, cutoff_1)
        plt.tight_layout()
        plt.savefig('x_images/plots/lc_val_' + str(model_name) + '.pdf')
        plt.show()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        plt.plot(time_long, emb_long, MarkerSize=3, color=cmap(14 / 20), marker='', alpha=1,
                 label='Random Initialization',
                 linewidth=2.,
                 linestyle=':')

        plt.plot(time_pre, emb_pre, MarkerSize=3, color=cmap(0.), marker='', alpha=1,
                 label='Pretrained Model', linewidth=2.,
                 linestyle=':')

        plt.plot(time_long, cel_long, MarkerSize=3, color=cmap(14 / 20), marker='', alpha=1,
                 label='Random Initialization',
                 linewidth=2.,
                 linestyle='-')

        plt.plot(time_pre, cel_pre, MarkerSize=3, color=cmap(0.), marker='', alpha=1,
                 label='Pretrained Model', linewidth=2.,
                 linestyle='-')

        random_patch = mpatches.Patch(color=cmap(14/20), label='Random Initialization')
        pre_trained_patch = mpatches.Patch(color=cmap(0.), label='Pretrained Model')

        emb_line = mlines.Line2D([], [], color='black', marker='', linestyle=':',
                                  markersize=15, label='Embedding Loss')
        cel_line = mlines.Line2D([], [], color='black', marker='', linestyle='-',
                                  markersize=15, label='Cross Entropy Loss')

        ax.legend(handles=[random_patch, pre_trained_patch, emb_line, cel_line])

        plt.title(str(model_name).upper() + ': Training Loss vs. Time [h]')
        plt.xlabel('Time [h]')
        plt.ylabel('Model Training Loss')

        if cutoff_2 != 0:
            plt.xlim(0, cutoff_2)
        plt.tight_layout()
        plt.savefig('x_images/plots/lc_' + str(model_name) + '.pdf')
        plt.show()

    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        cmap = plt.cm.get_cmap('tab20b')

        time_val_long = (d_val_emb_long[:, 0] - d_val_emb_long[0, 0]) / 3600
        time_long = (d_emb_long[:, 0] - d_emb_long[0, 0]) / 3600
        time_val_pre = (d_val_emb_pre[:, 0] - d_val_emb_pre[0, 0]) / 3600
        time_pre = (d_emb_pre[:, 0] - d_emb_pre[0, 0]) / 3600

        time_val_long, val_emb_long = zip(
            *sorted(zip(time_val_long, val_emb_long), key=lambda time_val_long: time_val_long[0]))
        time_long, emb_long = zip(
            *sorted(zip(time_long, emb_long), key=lambda time_long: time_long[0]))
        time_val_pre, val_emb_pre = zip(
            *sorted(zip(time_val_pre, val_emb_pre), key=lambda time_val_pre: time_val_pre[0]))
        time_pre, emb_pre = zip(
            *sorted(zip(time_pre, emb_pre), key=lambda time_pre: time_pre[0]))

        plt.plot(time_val_long, val_emb_long, MarkerSize=3, color=cmap(14/20), marker='', alpha=1,
                 label='Random Initialization',
                 linewidth=2.,
                 linestyle=':')

        plt.plot(time_val_pre, val_emb_pre, MarkerSize=3, color=cmap(0.), marker='', alpha=1,
                 label='Pretrained Model', linewidth=2.,
                 linestyle=':')

        plt.title(str(model_name).upper() + ': Validation Loss vs. Time [h]')
        plt.xlabel('Time [h]')
        plt.ylabel('Model Validation Loss Offset')

        random_patch = mpatches.Patch(color=cmap(14/20), label='Random Initialization')
        pre_trained_patch = mpatches.Patch(color=cmap(0.), label='Pretrained Model')

        emb_line = mlines.Line2D([], [], color='black', marker='', linestyle=':',
                                  markersize=15, label='Embedding Loss')

        ax.legend(handles=[random_patch, pre_trained_patch, emb_line])

        if cutoff_1 != 0:
            plt.xlim(0, cutoff_1)
        plt.tight_layout()
        plt.savefig('x_images/plots/lc_val_' + str(model_name) + '.pdf')
        plt.show()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        plt.plot(time_long, emb_long, MarkerSize=3, color=cmap(14/20), marker='', alpha=1,
                 label='Random Initialization',
                 linewidth=2.,
                 linestyle=':')

        plt.plot(time_pre, emb_pre, MarkerSize=3, color=cmap(0.), marker='', alpha=1,
                 label='Pretrained Model', linewidth=2.,
                 linestyle=':')

        plt.title(str(model_name).upper() + ': Training Loss vs. Time [h]')
        plt.xlabel('Time [h]')
        plt.ylabel('Model Training Loss Offset to [0, 1]')

        random_patch = mpatches.Patch(color=cmap(14/20), label='Random Initialization')
        pre_trained_patch = mpatches.Patch(color=cmap(0.), label='Pretrained Model')

        emb_line = mlines.Line2D([], [], color='black', marker='', linestyle=':',
                                  markersize=15, label='Embedding Loss')

        ax.legend(handles=[random_patch, pre_trained_patch, emb_line])

        if cutoff_2 != 0:
            plt.xlim(0, cutoff_2)
        plt.tight_layout()
        plt.savefig('x_images/plots/lc_' + str(model_name) + '.pdf')
        plt.show()


def plot_iter_curve(figsize, cutoff_1=0, cutoff_2=0):
    time_val_1, comb_val_1, time_1, comb_1 = prepare_lc_plot(1)
    time_val_2, comb_val_2, time_2, comb_2 = prepare_lc_plot(2)
    time_val_3, comb_val_3, time_3, comb_3 = prepare_lc_plot(3)
    time_val_5, comb_val_5, time_5, comb_5 = prepare_lc_plot(5)
    time_val_10, comb_val_10, time_10, comb_10 = prepare_lc_plot(10)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cmap = plt.cm.get_cmap('tab20b')

    print(time_val_1)

    plt.plot(time_val_1, comb_val_1, MarkerSize=3, color=cmap(0.), marker='o', alpha=0.9,
             label='No. of Iterations = 1',
             linewidth=3.,
             linestyle='-')
    plt.plot(time_val_2, comb_val_2, MarkerSize=3, color=cmap(4 / 20), marker='o', alpha=0.9,
             label='No. of Iterations = 2',
             linewidth=3.,
             linestyle='-')
    plt.plot(time_val_3, comb_val_3, MarkerSize=3, color=cmap(13 / 20), marker='o', alpha=0.9,
             label='No. of Iterations = 3',
             linewidth=3.,
             linestyle='-')
    plt.plot(time_val_5, comb_val_5, MarkerSize=3, color=cmap(11 / 20), marker='^', alpha=1.,
             label='No. of Iterations = 5',
             linewidth=2.,
             linestyle=':')
    plt.plot(time_val_10, comb_val_10, MarkerSize=3, color=cmap(17 / 20), marker='^', alpha=1.,
             label='No. of Iterations = 10',
             linewidth=2.,
             linestyle=':')

    plt.title('NOAH With Different Iterations: Validation Loss vs. Number of Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Model Validation Loss Offset to [0, 1]')

    if cutoff_1 != 0:
        plt.xlim(0, cutoff_1)

    ax.legend()
    plt.tight_layout()
    plt.savefig('x_images/plots/lc_val_iter.pdf')
    plt.show()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    plt.plot(time_1, comb_1, MarkerSize=5, color=cmap(0.), marker='o', alpha=0.5,
             label='No. of Iterations = 1',
             linewidth=1.5,
             linestyle=' ', markevery=3)
    plt.plot(time_2, comb_2, MarkerSize=5, color=cmap(4 / 20), marker='s', alpha=0.5,
             label='No. of Iterations = 2',
             linewidth=1.5,
             linestyle=' ', markevery=3)
    plt.plot(time_3, comb_3, MarkerSize=5, color=cmap(13/20), marker='x', alpha=0.8,
             label='No. of Iterations = 3',
             linewidth=1.5,
             linestyle=' ', markevery=3)
    plt.plot(time_5, comb_5, MarkerSize=5, color=cmap(11/20), marker='^', alpha=1.,
             label='No. of Iterations = 5',
             linewidth=1.5,
             linestyle=' ', markevery=5)
    plt.plot(time_10, comb_10, MarkerSize=3, color=cmap(17/20), marker='^', alpha=.5,
             label='No. of Iterations = 10',
             linewidth=1.5,
             linestyle=':', markevery=5)

    plt.title('Model With Different Iterations: Training Loss vs. Number of Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Model Training Loss Offset to [0, 1]')

    if cutoff_2 != 0:
        plt.xlim(0, cutoff_2)
    ax.legend(loc=('upper center'))
    plt.tight_layout()
    plt.savefig('x_images/plots/lc_iter.pdf')
    plt.show()

def prepare_lc_plot(iter_nb):
    d_val_emb_1 = np.array(data.read_from_json('data/learning_curves/iter_' + str(iter_nb) + '_val_emb.json'))
    d_emb_1 = np.array(data.read_from_json('data/learning_curves/iter_' + str(iter_nb) + '_emb.json'))

    d_val_emb_1 = np.where(np.isnan(d_val_emb_1) == 1, 0., d_val_emb_1)
    d_emb_1 = np.where(np.isnan(d_emb_1) == 1, 0., d_emb_1)

    val_emb_1 = d_val_emb_1[:, 2]
    emb_1 = d_emb_1[:, 2]

    val_emb_1 = val_emb_1 / np.linalg.norm(val_emb_1)
    emb_1 = emb_1 / np.linalg.norm(emb_1)

    d_val_cel_1 = np.array(data.read_from_json('data/learning_curves/iter_' + str(iter_nb) + '_val_cel.json'))
    d_cel_1 = np.array(data.read_from_json('data/learning_curves/iter_' + str(iter_nb) + '_cel.json'))

    d_val_cel_1 = np.where(np.isnan(d_val_cel_1) == 1, 0., d_val_cel_1)
    d_cel_1 = np.where(np.isnan(d_cel_1) == 1, 0., d_cel_1)

    time_val_1 = (d_val_cel_1[:, 0] - d_val_cel_1[0, 0]) / 3600
    time_1 = (d_cel_1[:, 0] - d_cel_1[0, 0]) / 3600

    val_cel_1 = d_val_cel_1[:, 2]
    cel_1 = d_cel_1[:, 2]

    val_cel_1 = val_cel_1 / np.linalg.norm(val_cel_1)
    cel_1 = cel_1 / np.linalg.norm(cel_1)

    comb_val_1 = (val_cel_1 * val_emb_1) / (val_cel_1 + val_emb_1)
    comb_1 = (cel_1 * emb_1) / (cel_1 + emb_1)

    comb_val_1 = h.zero_to_one(comb_val_1)
    comb_1 = h.zero_to_one(comb_1)

    training_steps = np.linspace(0, 50, len(comb_1))
    val_steps = np.linspace(0, 50, len(comb_val_1))

    val_steps, comb_val_1 = zip(
        *sorted(zip(val_steps, comb_val_1), key=lambda val_steps: val_steps[0]))
    training_steps, comb_1 = zip(
        *sorted(zip(training_steps, comb_1), key=lambda training_steps: training_steps[0]))

    return val_steps, comb_val_1, training_steps, comb_1



def plot_iter_curve_2(figsize, cutoff_1=0, cutoff_2=0):
    time_val_1, emb_val_1, cel_val_1, time_1, emb_1, cel_1 = prepare_lc_plot_2(1)
    time_val_2, emb_val_2, cel_val_2, time_2, emb_2, cel_2 = prepare_lc_plot_2(2)
    time_val_3, emb_val_3, cel_val_3, time_3, emb_3, cel_3 = prepare_lc_plot_2(3)
    time_val_5, emb_val_5, cel_val_5, time_5, emb_5, cel_5 = prepare_lc_plot_2(5)
    time_val_10, emb_val_10, cel_val_10, time_10, emb_10, cel_10 = prepare_lc_plot_2(10)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cmap = plt.cm.get_cmap('tab20b')

    print(time_val_1)

    plt.plot(time_val_1, emb_val_1, MarkerSize=3, color=cmap(0.), marker='o', alpha=0.9,
             label='No. of Iterations = 1',
             linewidth=3.,
             linestyle='-')
    plt.plot(time_val_2, emb_val_2, MarkerSize=3, color=cmap(4 / 20), marker='o', alpha=0.9,
             label='No. of Iterations = 2',
             linewidth=3.,
             linestyle='-')
    plt.plot(time_val_3, emb_val_3, MarkerSize=3, color=cmap(13 / 20), marker='o', alpha=0.9,
             label='No. of Iterations = 3',
             linewidth=3.,
             linestyle='-')
    plt.plot(time_val_5, emb_val_5, MarkerSize=3, color=cmap(11 / 20), marker='^', alpha=1.,
             label='No. of Iterations = 5',
             linewidth=2.,
             linestyle='-')
    plt.plot(time_val_10, emb_val_10, MarkerSize=3, color=cmap(17 / 20), marker='^', alpha=1.,
             label='No. of Iterations = 10',
             linewidth=2.,
             linestyle='-')

    plt.title('NOAH With Different Iterations: Validation Embedding Loss vs. Number of Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Model Validation Loss')

    if cutoff_1 != 0:
        plt.xlim(0, cutoff_1)

    ax.legend()
    plt.tight_layout()
    plt.savefig('x_images/plots/lc_val_emb_iter.pdf')
    plt.show()


    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cmap = plt.cm.get_cmap('tab20b')

    print(time_val_1)

    plt.plot(time_val_1, cel_val_1, MarkerSize=3, color=cmap(0.), marker='o', alpha=0.9,
             label='No. of Iterations = 1',
             linewidth=3.,
             linestyle='-')
    plt.plot(time_val_2, cel_val_2, MarkerSize=3, color=cmap(4 / 20), marker='o', alpha=0.9,
             label='No. of Iterations = 2',
             linewidth=3.,
             linestyle='-')
    plt.plot(time_val_3, cel_val_3, MarkerSize=3, color=cmap(13 / 20), marker='o', alpha=0.9,
             label='No. of Iterations = 3',
             linewidth=3.,
             linestyle='-')
    plt.plot(time_val_5, cel_val_5, MarkerSize=3, color=cmap(11 / 20), marker='^', alpha=1.,
             label='No. of Iterations = 5',
             linewidth=2.,
             linestyle='-')
    plt.plot(time_val_10, cel_val_10, MarkerSize=3, color=cmap(17 / 20), marker='^', alpha=1.,
             label='No. of Iterations = 10',
             linewidth=2.,
             linestyle='-')

    plt.title('NOAH With Different Iterations: Validation Cross Entropy Loss vs. Number of Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Model Validation Loss')

    if cutoff_1 != 0:
        plt.xlim(0, cutoff_1)

    ax.legend()
    plt.tight_layout()
    plt.savefig('x_images/plots/lc_val_cel_iter.pdf')
    plt.show()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    plt.plot(time_1, emb_1, MarkerSize=5, color=cmap(0.), marker='o', alpha=0.5,
             label='No. of Iterations = 1',
             linewidth=1.5,
             linestyle=' ', markevery=3)
    plt.plot(time_2, emb_2, MarkerSize=5, color=cmap(4 / 20), marker='s', alpha=0.5,
             label='No. of Iterations = 2',
             linewidth=1.5,
             linestyle=' ', markevery=3)
    plt.plot(time_3, emb_3, MarkerSize=5, color=cmap(13/20), marker='x', alpha=0.8,
             label='No. of Iterations = 3',
             linewidth=1.5,
             linestyle=' ', markevery=3)
    plt.plot(time_5, emb_5, MarkerSize=5, color=cmap(11/20), marker='^', alpha=1.,
             label='No. of Iterations = 5',
             linewidth=1.5,
             linestyle=' ', markevery=5)
    plt.plot(time_10, emb_10, MarkerSize=3, color=cmap(17/20), marker='^', alpha=.5,
             label='No. of Iterations = 10',
             linewidth=1.5,
             linestyle='', markevery=5)

    plt.title('Model With Different Iterations: Training Embedding Loss vs. Number of Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Model Training Loss')

    if cutoff_2 != 0:
        plt.xlim(0, cutoff_2)
    ax.legend(loc=('upper center'))
    plt.tight_layout()
    plt.savefig('x_images/plots/lc_emb_iter.pdf')
    plt.show()


    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    plt.plot(time_1, cel_1, MarkerSize=5, color=cmap(0.), marker='o', alpha=0.5,
             label='No. of Iterations = 1',
             linewidth=1.5,
             linestyle=' ', markevery=3)
    plt.plot(time_2, cel_2, MarkerSize=5, color=cmap(4 / 20), marker='s', alpha=0.5,
             label='No. of Iterations = 2',
             linewidth=1.5,
             linestyle=' ', markevery=3)
    plt.plot(time_3, cel_3, MarkerSize=5, color=cmap(13/20), marker='x', alpha=0.8,
             label='No. of Iterations = 3',
             linewidth=1.5,
             linestyle=' ', markevery=3)
    plt.plot(time_5, cel_5, MarkerSize=5, color=cmap(11/20), marker='^', alpha=1.,
             label='No. of Iterations = 5',
             linewidth=1.5,
             linestyle=' ', markevery=5)
    plt.plot(time_10, cel_10, MarkerSize=3, color=cmap(17/20), marker='^', alpha=.5,
             label='No. of Iterations = 10',
             linewidth=1.5,
             linestyle='', markevery=5)

    plt.title('Model With Different Iterations: Training Cross Entropy Loss vs. Number of Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Model Training Loss')

    if cutoff_2 != 0:
        plt.xlim(0, cutoff_2)
    ax.legend(loc=('upper center'))
    plt.tight_layout()
    plt.savefig('x_images/plots/lc_cel_iter.pdf')
    plt.show()



def prepare_lc_plot_2(iter_nb):
    d_val_emb_1 = np.array(data.read_from_json('data/learning_curves/iter_' + str(iter_nb) + '_val_emb.json'))
    d_emb_1 = np.array(data.read_from_json('data/learning_curves/iter_' + str(iter_nb) + '_emb.json'))

    d_val_emb_1 = np.where(np.isnan(d_val_emb_1) == 1, 0., d_val_emb_1)
    d_emb_1 = np.where(np.isnan(d_emb_1) == 1, 0., d_emb_1)

    val_emb_1 = d_val_emb_1[:, 2]
    emb_1 = d_emb_1[:, 2]

    d_val_cel_1 = np.array(data.read_from_json('data/learning_curves/iter_' + str(iter_nb) + '_val_cel.json'))
    d_cel_1 = np.array(data.read_from_json('data/learning_curves/iter_' + str(iter_nb) + '_cel.json'))

    d_val_cel_1 = np.where(np.isnan(d_val_cel_1) == 1, 0., d_val_cel_1)
    d_cel_1 = np.where(np.isnan(d_cel_1) == 1, 0., d_cel_1)

    time_val_1 = (d_val_cel_1[:, 0] - d_val_cel_1[0, 0]) / 3600
    time_1 = (d_cel_1[:, 0] - d_cel_1[0, 0]) / 3600

    val_cel_1 = d_val_cel_1[:, 2]
    cel_1 = d_cel_1[:, 2]

    training_steps = np.linspace(0, 50, len(emb_1))
    val_steps = np.linspace(0, 50, len(val_emb_1))

    val_steps, emb_val_1 = zip(
        *sorted(zip(val_steps, val_emb_1), key=lambda val_steps: val_steps[0]))
    training_steps, emb_1 = zip(
        *sorted(zip(training_steps, emb_1), key=lambda training_steps: training_steps[0]))
    val_steps, cel_val_1 = zip(
        *sorted(zip(val_steps, val_cel_1), key=lambda val_steps: val_steps[0]))
    training_steps, cel_1 = zip(
        *sorted(zip(training_steps, cel_1), key=lambda training_steps: training_steps[0]))

    return val_steps, emb_val_1, cel_val_1, training_steps, emb_1, cel_1
