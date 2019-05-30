import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul as mm
import numpy as np

import config as c

def embedding_loss(embedding_matrix, labels, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method that exhaustively calculates the embedding loss of an embedding when given the labels
    :param embedding_matrix: matrix with the predicted embeddings
    :param labels: torch tensor with the ground truth
    :param dtype: dtype of the tensors
    :param device: device which cuda uses
    :return: the total loss
    """

    embedding_dim = embedding_matrix.size(1)
    pic_dim = embedding_matrix.size()[2:4]

    # calculating similarity matrix
    similarity_matrix = torch.zeros(pic_dim[0] * pic_dim[1], pic_dim[0] * pic_dim[1], dtype=dtype, device=device)
    labels_matrix = torch.zeros(pic_dim[0] * pic_dim[1], pic_dim[0] * pic_dim[1], dtype=dtype, device=device)
    pre_weights_matrix = labels
    weights_matrix = torch.zeros_like(labels_matrix)

    # for w in range(labels.size(0)):
    #     for v in range(labels.size(1)):
    #         pre_weights_matrix[w, v] = torch.div(torch.tensor(1, dtype=dtype, device=device),
    #                                              ((labels == labels[w, v]).nonzero()).size(0))

    # calculate pre-weighting matrix
    unique_labels = torch.unique(labels, sorted=True)
    'see if enumerate works correctly here'
    for _, l in enumerate(unique_labels):
        pre_weights_matrix = torch.where(pre_weights_matrix == l, torch.div(torch.tensor(
            1, dtype=dtype, device=device), ((labels == l).nonzero()).size(0)), pre_weights_matrix)
    del unique_labels


    for n in range(embedding_matrix[0, :, :].size(0) * embedding_matrix[0, :, :].size(1)):
        rolling_matrix = torch.roll(embedding_matrix.view(embedding_dim, -1), shifts=n, dims=1)
        rolling_labels = torch.roll(labels.view(-1), shifts=n, dims=0)
        print(n)
        for p in range(rolling_matrix.size(1) - n):
            # if same pixel do not calculate similarity (diagonal zeroes)
            if n == p:
                pass
            # print('similarity matrix', similarity_matrix.size())
            # print('embedding matrix', embedding_matrix.view(embedding_dim, -1).size())
            # [print('rolling matrix', rolling_matrix.size())]
            similarity_matrix[n, p] = cos_similarity(embedding_matrix.view(embedding_dim, -1)[:, p],
                                                     rolling_matrix[:, p])
            if labels.view(-1)[n] == rolling_labels[p]:
                labels_matrix[n, p] = 1
            elif labels.view(-1)[n] != rolling_labels[p]:
                labels_matrix[n, p] = -1

            # setting weight matrix
            weights_matrix[n, p] = torch.mul(pre_weights_matrix.view(-1)[n], pre_weights_matrix.view(-1)[p])

    del rolling_matrix, rolling_labels

    indices_positive = (labels_matrix == 1).nonzero()
    indices_negative = (labels_matrix == -1).nonzero()

    loss = torch.zeros(pic_dim[0] * pic_dim[1], pic_dim[0] * pic_dim[1], dtype=dtype, device=device)
    loss[indices_positive] = torch.sub(1, similarity_matrix[indices_positive])
    loss[indices_negative] = torch.where(similarity_matrix[indices_negative] - c.embedding_loss['margin'] >= 0,
                                         similarity_matrix[indices_negative] - c.embedding_loss['margin'], torch.tensor(
            0, device=device, dtype=dtype))

    del indices_positive, indices_negative

    loss = torch.mul(1 / (pic_dim[0] * pic_dim[1]), torch.sum(torch.mul(weights_matrix, loss)))

    return loss