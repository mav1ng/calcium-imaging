import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul as mm
import numpy as np

import config as c


def embedding_loss(embedding_matrix, labels, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method that exhaustively calculates the embedding loss of an embedding when given the labels
    :param embedding_matrix: matrix with the predicted embeddings C x PixelX x PixelY
    :param labels: torch tensor with the ground truth
    :param dtype: dtype of the tensors
    :param device: device which cuda uses
    :return: the total loss
    """

    pic_dim = embedding_matrix.size()[1:3]

    similarity_matrix = compute_similarity(embedding_matrix=embedding_matrix, dtype=dtype, device=device)
    weights = compute_weight_matrix(compute_pre_weight_matrix(labels_matrix=labels, dtype=dtype, device=device),
                                    dtype=dtype, device=device)
    label_pairs = compute_label_pair(label_matrix=labels, dtype=dtype, device=device)

    loss = torch.zeros(pic_dim[0] * pic_dim[1], pic_dim[0] * pic_dim[1], dtype=dtype, device=device)
    loss = torch.where(label_pairs == 1, torch.sub(1., similarity_matrix), loss)
    loss = torch.where(label_pairs == -1, torch.where(similarity_matrix - c.embedding_loss['margin'] >= 0,
                                                      similarity_matrix - c.embedding_loss['margin'], torch.tensor(
            0, device=device, dtype=dtype)), loss)

    loss = torch.mul(1. / (pic_dim[0] * pic_dim[1]), torch.sum(torch.mul(weights, loss)))

    return loss

def embedding_loss_new(embedding_matrix, labels, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method that exhaustively calculates the embedding loss of an embedding when given the labels
    :param embedding_matrix: matrix with the predicted embeddings C x PixelX x PixelY
    :param labels: torch tensor with the ground truth
    :param dtype: dtype of the tensors
    :param device: device which cuda uses
    :return: the total loss
    """

    pic_dim = embedding_matrix.size()[1:3]

    similarity_matrix = compute_similarity_new(embedding_matrix=embedding_matrix, dtype=dtype, device=device)
    weights = compute_weight_matrix_new(compute_pre_weight_matrix(labels_matrix=labels, dtype=dtype, device=device),
                                    dtype=dtype, device=device)
    label_pairs = compute_label_pair_new(label_matrix=labels, dtype=dtype, device=device)

    # loss = torch.zeros(pic_dim[0] * pic_dim[1], pic_dim[0] * pic_dim[1], dtype=dtype, device=device)
    loss = torch.where(label_pairs == 1, torch.sub(1., similarity_matrix), torch.tensor(0, device=device, dtype=dtype))
    loss = torch.where(label_pairs == -1, torch.where(similarity_matrix - c.embedding_loss['margin'] >= 0,
                                                      similarity_matrix - c.embedding_loss['margin'], torch.tensor(
            0, device=device, dtype=dtype)), loss)

    loss = torch.mul(1. / (pic_dim[0] * pic_dim[1]), torch.sum(torch.mul(weights, loss))) / 2.

    return loss



def compute_similarity(embedding_matrix, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method that computes the similarity matrix to a correspodning embedding matrix where the similarity
    is gives by 1/2 * (1 + (x_i.T * x_j)/(||x_1||*||x_y||))
    :param embedding_matrix: tensor C x PixelX x PixelY
    :return: tensor similarity matrix PixelX * PixelY x PixelX * PixelY, where it is a triangular matrix with zeroes
    from the diagonal and below because this would be repeating information and in the upper part of the matrix
    pairwise similarity values
    """

    embedding_dim = embedding_matrix.size(0)
    x = embedding_matrix.view(embedding_dim, -1)
    x_norm = x / x.norm(dim=0)[None, :]

    similarity_matrix = torch.mm(x_norm.transpose(0, 1), x_norm)

    sim_dim_x = similarity_matrix.size(0)
    sim_dim_y = similarity_matrix.size(1)

    similarity_matrix = torch.where(torch.tensor(np.tri(sim_dim_x, sim_dim_y), dtype=dtype, device=device) == 1.,
                                    torch.tensor(0, dtype=dtype, device=device), torch.mul(torch.add(similarity_matrix,
                                                                                                     1.), 1./2.))
    # del sim_dim_x, sim_dim_y, x, x_norm

    return similarity_matrix


def compute_similarity_new(embedding_matrix, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method that computes the similarity matrix to a correspodning embedding matrix where the similarity
    is gives by 1/2 * (1 + (x_i.T * x_j)/(||x_1||*||x_y||))
    :param embedding_matrix: tensor C x PixelX x PixelY
    :return: tensor similarity matrix PixelX * PixelY x PixelX * PixelY, where it is a triangular matrix with zeroes
    from the diagonal and below because this would be repeating information and in the upper part of the matrix
    pairwise similarity values
    """

    embedding_dim = embedding_matrix.size(0)
    x = embedding_matrix.view(embedding_dim, -1)
    x_norm = x / x.norm(dim=0)[None, :]

    r = torch.mm(x_norm.transpose(0, 1), x_norm) * 0.5 + 0.5

    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)

    D = (diag + diag.t() - 2*r)

    return D.sqrt()


def compute_label_pair(label_matrix, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method that computes whether a pixel pair is of the same label or not
    :param label_matrix: tensor matrix with the ground truth labels PixelX x PixelY
    :param dtype:
    :param device:
    :return: tensor label pair matrix PixelX * PixelY x PixelX * PixelY, where it is a triangular matrix with zeroes
    from the diagonal and below because this would be repeating information and in the upper part of the matrix 1 if
    positive pair, -1 if negative pair, 0 if repeating information
    """

    # +1 such that background has label != 0 such that following computation works
    x = torch.add(label_matrix.view(-1), 1.).view(1, -1)
    label_pair_matrix = torch.mm(x.transpose(0, 1), x)

    sim_dim_x = label_pair_matrix.size(0)
    sim_dim_y = label_pair_matrix.size(1)

    # 1 if positive pair, -1 if negative pair, 0 if repeating information
    label_pair_matrix = torch.where(
        torch.mm(x.transpose(0, 1), torch.ones_like(x, dtype=dtype, device=device)) == torch.sqrt(label_pair_matrix),
        torch.tensor(1, dtype=dtype, device=device), torch.tensor(-1, dtype=dtype, device=device))

    label_pair_matrix = torch.where(torch.tensor(np.tri(sim_dim_x, sim_dim_y), dtype=dtype, device=device) == 1,
                                    torch.tensor(0, dtype=dtype, device=device), label_pair_matrix)

    # del sim_dim_x, sim_dim_y, x

    return label_pair_matrix


def compute_label_pair_new(label_matrix, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method that computes whether a pixel pair is of the same label or not
    :param label_matrix: tensor matrix with the ground truth labels PixelX x PixelY
    :param dtype:
    :param device:
    :return: tensor label pair matrix PixelX * PixelY x PixelX * PixelY, where it is a triangular matrix with zeroes
    from the diagonal and below because this would be repeating information and in the upper part of the matrix 1 if
    positive pair, -1 if negative pair, 0 if repeating information
    """

    with torch.no_grad():
        # +1 such that background has label != 0 such that following computation works
        x = torch.add(label_matrix.view(-1), 1.).view(1, -1)
        label_pair_matrix = torch.mm(x.transpose(0, 1), x)

        sim_dim_x = label_pair_matrix.size(0)
        sim_dim_y = label_pair_matrix.size(1)

        # 1 if positive pair, -1 if negative pair, 0 if repeating information
        label_pair_matrix = torch.where(
            torch.mm(x.transpose(0, 1), torch.ones_like(x, dtype=dtype, device=device)) == torch.sqrt(label_pair_matrix),
            torch.tensor(1, dtype=dtype, device=device), torch.tensor(-1, dtype=dtype, device=device))
    return label_pair_matrix


def compute_pre_weight_matrix(labels_matrix, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method to compute the pre_weight_matrix used for the computation of the weight matrix
    :param labels_matrix: tensor matrix with the ground truth labels PixelX x PixelY
    :param dtype:
    :param device:
    :return: tensor PixelX x PixelY with weighted pixels for every label (1/number of pixels with the same label
    """
    with torch.no_grad():
        # calculate pre-weighting matrix
        pre_weights_matrix = labels_matrix
        unique_labels = torch.unique(labels_matrix, sorted=False)
        for _, l in enumerate(unique_labels):
            pre_weights_matrix = torch.where(pre_weights_matrix == l, torch.div(torch.tensor(
                1., dtype=dtype, device=device), float((labels_matrix == l).nonzero().size(0))), pre_weights_matrix)

    return pre_weights_matrix


def compute_weight_matrix(pre_weight_matrix, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method to compute the weight_matrix for the loss function
    :param pre_weight_matrix: tensor PixelX x PixelY with weighted pixels for every label (1/number of pixels with the
    same label
    :param dtype:
    :param device:
    :return: weight matrix PixelX * PixelY x PixelX * PixelY, where it is a triangular matrix with zeroes
    from the diagonal and below because this would be repeating information and in the upper part of the matrix the
    weight pairs for the computation of the loss function
    """

    x = pre_weight_matrix.view(1, -1)
    weight_matrix = torch.mm(x.transpose(0, 1), x)

    sim_dim_x = weight_matrix.size(0)
    sim_dim_y = weight_matrix.size(1)

    weight_matrix = torch.where(torch.tensor(np.tri(sim_dim_x, sim_dim_y), dtype=dtype, device=device) == 1,
                                torch.tensor(0, dtype=dtype, device=device), weight_matrix)

    # del sim_dim_x, sim_dim_y, x

    return weight_matrix


def compute_weight_matrix_new(pre_weight_matrix, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method to compute the weight_matrix for the loss function
    :param pre_weight_matrix: tensor PixelX x PixelY with weighted pixels for every label (1/number of pixels with the
    same label
    :param dtype:
    :param device:
    :return: weight matrix PixelX * PixelY x PixelX * PixelY, where it is a triangular matrix with zeroes
    from the diagonal and below because this would be repeating information and in the upper part of the matrix the
    weight pairs for the computation of the loss function
    """

    x = pre_weight_matrix.view(1, -1)
    weight_matrix = torch.mm(x.transpose(0, 1), x)

    return weight_matrix


def get_embedding_loss(embedding_list, labels, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method to compute the accumulated loss after the several MS iterations
    :param embedding_list: tensor dimension I x D x W x H
    :param labels: Ground Truth of Labels
    :param dtype:
    :param device:
    :return: Tensor, returns the accumulated loss
    """
    loss = torch.tensor(0., dtype=dtype, device=device,requires_grad=True)
    for i in range(0, embedding_list.size(0)):
        loss = torch.add(loss, embedding_loss(embedding_matrix=embedding_list[i, :, :, :], labels=labels, dtype=dtype,
                                              device=device))
    # weighting the loss depending on how many neurons are in the picture
    # loss = torch.mul(labels.size(0) * labels.size(1)/(labels.nonzero().size(0)), loss)
    return loss


def get_embedding_loss_new(embedding_list, labels, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method to compute the accumulated loss after the several MS iterations
    :param embedding_list: tensor dimension I x D x W x H
    :param labels: Ground Truth of Labels
    :param dtype:
    :param device:
    :return: Tensor, returns the accumulated loss
    """
    loss = torch.tensor(0., dtype=dtype, device=device,requires_grad=True)
    for i in range(0, embedding_list.size(0)):
        loss = torch.add(loss, embedding_loss_new(embedding_matrix=embedding_list[i, :, :, :], labels=labels, dtype=dtype,
                                              device=device))
    # weighting the loss depending on how many neurons are in the picture
    # loss = torch.mul(labels.size(0) * labels.size(1)/(labels.nonzero().size(0)), loss)
    return loss


def get_batch_embedding_loss(embedding_list, labels_list, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method to compute the accumulated loss after the several MS iterations
    :param embedding_list: tensor dimension B x I x D x W x H
    :param labels_list: Ground Truth of Labels of dimension B x W x H
    :param dtype:
    :param device:
    :return: Tensor, returns the accumulated loss over a batch
    """
    loss = torch.tensor(0., dtype=dtype, device=device, requires_grad=True)
    for b in range(embedding_list.size(0)):
        loss = torch.add(loss,
                         get_embedding_loss(embedding_list[b], labels=labels_list[b], dtype=dtype,
                                            device=device))
    return loss


def get_batch_embedding_loss_new(embedding_list, labels_list, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method to compute the accumulated loss after the several MS iterations
    :param embedding_list: tensor dimension B x I x D x W x H
    :param labels_list: Ground Truth of Labels of dimension B x W x H
    :param dtype:
    :param device:
    :return: Tensor, returns the accumulated loss over a batch
    """
    loss = torch.tensor(0., dtype=dtype, device=device, requires_grad=True)
    for b in range(embedding_list.size(0)):
        loss = torch.add(loss,
                         get_embedding_loss_new(embedding_list[b], labels=labels_list[b], dtype=dtype,
                                            device=device))
    return loss


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