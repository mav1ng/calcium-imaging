import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering


def label_emb_sl(data, th):
    """
    Method that labels the clustered embeddings based on sklearn AgglomerativeClustering
    :param data: numpy array N x D, N number of pixels, D embedding dim
    :param th: threshold radius of the sk learn nearest neighbour ball
    :return: N x 1 labelled pixels
    """
    d = data.cpu().numpy()
    clustering = AgglomerativeClustering(linkage='single', n_clusters=None, distance_threshold=th).fit(d)
    print('Found ' + str(clustering.n_clusters_) + ' clusters.')
    return clustering.labels_


def label_embeddings(data, th):
    """
    Method that labels the clustered embeddings based on sklearn
    :param data: numpy array N x D, N number of pixels, D embedding dim
    :param th: threshold radius of the sk learn nearest neighbour ball
    :return: N x 1 labelled pixels
    """
    d = data.cpu().numpy()
    out = np.zeros((data.shape[0], 1))
    label = 0
    neigh = NearestNeighbors(radius=th)
    neigh.fit(d)
    while d.shape[0] > 0:
        seed = np.random.randint(0, d.shape[0])
        rng = neigh.radius_neighbors(d[seed].reshape(1, -1))
        ind = np.asarray(rng[1][0])
        out[ind] = label
        out[seed] = label
        ind = np.append(ind, seed)
        d = np.delete(d, ind, axis=0)
        label = label + 1
    return out


def cluster_kmean(data, tol=0.001, max_iter=10000):
    """
    Method that clusters the data via k means clustering. Input should be N x D, N number of pixels, D dimension of
    embedding
    :param data:
    :param tol:
    :param max_iter:
    :return: returns labels and center means
    """
    opt_nb, opt_wss = 0, np.inf
    for i in range(2, 20):
        print(i)
        clusters_index, centers = lloyd(data, i, tol=0.001, max_iter=10000)
        curwss = wss(data, clusters_index, centers)
        if opt_wss > curwss:
            opt_wss = curwss
            opt_nb = i
        print(opt_nb, opt_wss)
    clusters_index, centers = lloyd(data, opt_nb, tol=0.001, max_iter=10000)
    return clusters_index, centers

"""
Mainly Imported from https://github.com/overshiki/kmeans_pytorch"""

def forgy(X, n_clusters):
    _len = len(X)
    indices = np.random.choice(_len, n_clusters)
    initial_state = X[indices]
    return initial_state


def lloyd(X, n_clusters, tol=1e-4, max_iter=1000):
    try:
        X = torch.from_numpy(X).float().cuda()
    except TypeError:
        Y = X.float().cuda()

    initial_state = forgy(X, n_clusters)
    counter = 0

    while True:
        dis = pairwise_distance(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
        counter = counter + 1

        if center_shift ** 2 < tol or counter >= max_iter:
            break

    return choice_cluster.cpu().numpy(), initial_state.cpu().numpy()

def pairwise_distance(data1, data2=None):
    r'''
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    '''
    if data2 is None:
        data2 = data1

    data1, data2 = data1.cuda(), data2.cuda()

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def wss(data, cluster_indices, cluster_means):
    try:
        X = torch.from_numpy(data).float().cuda()
    except TypeError:
        X = data.float().cuda()
    try:
        ci = torch.from_numpy(cluster_indices).float().cuda()
    except TypeError:
        ci = cluster_indices.float().cuda()
    try:
        cm = torch.from_numpy(cluster_means).float().cuda()
    except TypeError:
        cm = cluster_means.float().cuda()

    wss_tot = 0.
    for i, cluster in enumerate(cluster_means):
        indices = (ci == i).nonzero()
        wss_tot = wss_tot + torch.sum((X[indices] - cm[i]) ** 2)
    return wss_tot



# def group_pairwise(X, groups, device=0, fun=lambda r, c: pairwise_distance(r, c).cpu()):
#     group_dict = {}
#     for group_index_r, group_r in enumerate(groups):
#         for group_index_c, group_c in enumerate(groups):
#             R, C = X[group_r], X[group_c]
#             if device != -1:
#                 R = R.cuda(device)
#                 C = C.cuda(device)
#             group_dict[(group_index_r, group_index_c)] = fun(R, C)
#
#     return group_dict

