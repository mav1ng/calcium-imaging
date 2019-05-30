import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul as mm
import numpy as np

import config as c


def get_conv_layer(num_input, num_output):
    """
    method to get the basic 3x3 blocks of the U-Net network
    :param num_input:
    :param num_output:
    :return: Sequential() of the basic block
    """
    return nn.Sequential(
        nn.Conv2d(num_input, num_output, 3, stride=1, padding=(1, 1)),
        nn.BatchNorm2d(num_output, momentum=0.5),
    )


def get_up_layer(num_input, num_output):
    return nn.Sequential(
        nn.ConvTranspose2d(num_input, num_output, kernel_size=2, stride=2),
        nn.BatchNorm2d(num_output, momentum=0.5),
    )


def normalize(input_matrix):
    return F.normalize(input_matrix, p=2, dim=2)


def cos_similarity(vec1, vec2):
    return 1 / 2 * (1 + torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2)))


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.input_channels = c.UNet['input_channels']
        self.embedding_dim = c.UNet['embedding_dim']
        self.dropout_rate = c.UNet['dropout_rate']

        self.conv_layer_1 = get_conv_layer(self.input_channels, self.embedding_dim)
        self.conv_layer_2 = get_conv_layer(self.embedding_dim, self.embedding_dim)

        self.max_pool_2D_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_3 = get_conv_layer(self.embedding_dim, self.embedding_dim * 2)
        self.conv_layer_4 = get_conv_layer(self.embedding_dim * 2, self.embedding_dim * 2)
        self.dropout_1 = nn.Dropout(p=self.dropout_rate)

        self.max_pool_2D_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_5 = get_conv_layer(self.embedding_dim * 2, self.embedding_dim * 4)
        self.conv_layer_6 = get_conv_layer(self.embedding_dim * 4, self.embedding_dim * 4)
        self.dropout_2 = nn.Dropout(p=self.dropout_rate * 2)

        self.max_pool_2D_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_7 = get_conv_layer(self.embedding_dim * 4, self.embedding_dim * 8)
        self.conv_layer_8 = get_conv_layer(self.embedding_dim * 8, self.embedding_dim * 8)
        self.dropout_3 = nn.Dropout(p=self.dropout_rate * 2)

        self.max_pool_2D_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_9 = get_conv_layer(self.embedding_dim * 8, self.embedding_dim * 16)
        self.conv_layer_10 = get_conv_layer(self.embedding_dim * 16, self.embedding_dim * 16)
        self.up_layer_1 = get_up_layer(self.embedding_dim * 16, self.embedding_dim * 8)
        self.dropout_4 = nn.Dropout(p=self.dropout_rate * 2)

        self.conv_layer_11 = get_conv_layer(self.embedding_dim * 16, self.embedding_dim * 8)
        self.conv_layer_12 = get_conv_layer(self.embedding_dim * 8, self.embedding_dim * 8)
        self.up_layer_2 = get_up_layer(self.embedding_dim * 8, self.embedding_dim * 4)
        self.dropout_5 = nn.Dropout(p=self.dropout_rate * 2)

        self.conv_layer_13 = get_conv_layer(self.embedding_dim * 8, self.embedding_dim * 4)
        self.conv_layer_14 = get_conv_layer(self.embedding_dim * 4, self.embedding_dim * 4)
        self.up_layer_3 = get_up_layer(self.embedding_dim * 4, self.embedding_dim * 2)
        self.dropout_6 = nn.Dropout(p=self.dropout_rate * 2)

        self.conv_layer_15 = get_conv_layer(self.embedding_dim * 4, self.embedding_dim * 2)
        self.conv_layer_16 = get_conv_layer(self.embedding_dim * 2, self.embedding_dim * 2)
        self.up_layer_4 = get_up_layer(self.embedding_dim * 2, self.embedding_dim)
        self.dropout_7 = nn.Dropout(p=self.dropout_rate)

        self.conv_layer_17 = get_conv_layer(self.embedding_dim * 2, self.embedding_dim)
        self.conv_layer_18 = get_conv_layer(self.embedding_dim, self.embedding_dim)
        self.conv_layer_end = nn.Conv2d(self.embedding_dim, self.embedding_dim, 1)

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = F.relu(x)
        x = self.conv_layer_2(x)
        x = F.relu(x)
        conc_in_1 = x  # used later when filters are concatenated

        x = self.max_pool_2D_1(x)
        x = self.conv_layer_3(x)
        x = F.relu(x)
        x = self.conv_layer_4(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        conc_in_2 = x

        x = self.max_pool_2D_2(x)
        x = self.conv_layer_5(x)
        x = F.relu(x)
        x = self.conv_layer_6(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        conc_in_3 = x

        x = self.max_pool_2D_3(x)
        x = self.conv_layer_7(x)
        x = F.relu(x)
        x = self.conv_layer_8(x)
        x = F.relu(x)
        x = self.dropout_3(x)
        conc_in_4 = x

        x = self.max_pool_2D_4(x)
        x = self.conv_layer_9(x)
        x = F.relu(x)
        x = self.conv_layer_10(x)
        x = F.relu(x)
        x = self.up_layer_1(x)
        x = F.relu(x)
        x = self.dropout_4(x)

        x = torch.cat((x, conc_in_4), dim=1)
        x = self.conv_layer_11(x)
        x = F.relu(x)
        x = self.conv_layer_12(x)
        x = F.relu(x)
        x = self.up_layer_2(x)
        x = F.relu(x)
        x = self.dropout_5(x)

        x = torch.cat((x, conc_in_3), dim=1)
        x = self.conv_layer_13(x)
        x = F.relu(x)
        x = self.conv_layer_14(x)
        x = F.relu(x)
        x = self.up_layer_3(x)
        x = F.relu(x)
        x = self.dropout_6(x)

        x = torch.cat((x, conc_in_2), dim=1)
        x = self.conv_layer_15(x)
        x = F.relu(x)
        x = self.conv_layer_16(x)
        x = F.relu(x)
        x = self.up_layer_4(x)
        x = F.relu(x)
        x = self.dropout_7(x)

        x = torch.cat((x, conc_in_1), dim=1)
        x = self.conv_layer_17(x)
        x = F.relu(x)
        x = self.conv_layer_18(x)
        x = F.relu(x)
        x = self.conv_layer_end(x)
        x = F.softmax(x, dim=-1)

        return x


class MS(nn.Module):

    def __init__(self):
        super(MS, self).__init__()

        self.embedding_dim = c.mean_shift['embedding_dim']
        # setting kernel bandwidth
        if c.mean_shift['kernel_bandwidth'] is not None:
            self.kernel_bandwidth = c.mean_shift['kernel_bandwidth']
        else:
            self.kernel_bandwidth = 1 / (1 - c.embedding_loss['margin']) / 3
        self.step_size = c.mean_shift['step_size']
        self.nb_iterations = c.mean_shift['nb_iterations']
        self.embeddings_list_tensor = torch.tensor([])
        self.nb_pixels = None  # to be defined when forward is called
        self.pic_res = None
        self.device = c.cuda['device']

    # x_in flattened image in D x N , N number of Pixels

    def forward(self, x_in):
        """
        :param x_in: flattened image in D x N , N number of Pixels
        :return: embeddings x_in mean shifted
        """
        with torch.no_grad():
            self.pic_res = x_in.size(3)

        x = x_in.view(self.embedding_dim, -1)
        print(x.size())

        with torch.no_grad():
            self.nb_pixels = x.size(1)
            print('input Mean Shift Block' + str(x.size()))

        for t in range(self.nb_iterations):
            x = x.view(-1, self.embedding_dim, self.nb_pixels)
            print('Mean Shift: ' + str(t) + ' First Iteration')
            # kernel_mat N x N , N number of pixels
            kernel_mat = torch.exp(torch.mul(self.kernel_bandwidth, mm(
                x[t, :, :].view(self.embedding_dim,
                                self.nb_pixels).t(), x[t, :, :].view(self.embedding_dim, self.nb_pixels))))
            # diag_mat N x N
            diag_mat = torch.diag(
                mm(kernel_mat.t(), torch.ones((self.nb_pixels, 1), device=self.device)).squeeze(dim=1), diagonal=0)

            x = torch.cat((x.view(-1, self.embedding_dim, self.pic_res, self.pic_res), mm(x[t, :, :],
                   torch.mul(self.step_size, mm(kernel_mat, torch.inverse(diag_mat))) +
                   torch.mul((1 - self.step_size), torch.eye(self.nb_pixels, self.nb_pixels, device=self.device))).view(
                1, self.embedding_dim, self.pic_res, self.pic_res)))



            '''WORKING HERE AT THE MOMENT'''
            '''NEW'''
            # self.embeddings_list_tensor = torch.cat((x.view(1, self.embedding_dim, self.pic_res, self.pic_res),
            #                                          self.embeddings_list_tensor.view(-1, self.embedding_dim,
            #                                                                           self.pic_res, self.pic_res)))
            '''OLD'''
            # with torch.no_grad():
            #     self.embeddings_list.append(x.view(self.embedding_dim, self.pic_res, self.pic_res))

        return x.view(-1, self.embedding_dim, self.pic_res, self.pic_res)


class UNetMS(nn.Module):
    def __init__(self):
        super(UNetMS, self).__init__()

        self.UNet = UNet()
        self.MS = MS()

    def forward(self, x):
        x = self.UNet(x)
        """Does the normalization really work that way?"""
        x = F.normalize(x, p=2, dim=1)
        x = self.MS(x)
        return x


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

    # indices_positive = (label_pairs == 1).nonzero()
    # indices_negative = (label_pairs == -1).nonzero()

    loss = torch.zeros(pic_dim[0] * pic_dim[1], pic_dim[0] * pic_dim[1], dtype=dtype, device=device)
    loss = torch.where(label_pairs == 1, torch.sub(1., similarity_matrix), loss)
    loss = torch.where(label_pairs == -1, torch.where(similarity_matrix - c.embedding_loss['margin'] >= 0,
                                                      similarity_matrix - c.embedding_loss['margin'], torch.tensor(
            0, device=device, dtype=dtype)), loss)

    # loss[indices_positive] = torch.sub(1., similarity_matrix[indices_positive])
    # loss[indices_negative] = torch.where(similarity_matrix[indices_negative] - c.embedding_loss['margin'] >= 0,
    #                                      similarity_matrix[indices_negative] - c.embedding_loss['margin'], torch.tensor(
    #         0, device=device, dtype=dtype))

    # del indices_positive, indices_negative

    loss = torch.mul(1 / (pic_dim[0] * pic_dim[1]), torch.sum(torch.mul(weights, loss)))

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

    similarity_matrix = torch.where(torch.tensor(np.tri(sim_dim_x, sim_dim_y), dtype=dtype, device=device) == 1,
                                    torch.tensor(0, dtype=dtype, device=device), torch.mul(torch.add(similarity_matrix,
                                                                                                     1.), 1 / 2))
    del sim_dim_x, sim_dim_y, x, x_norm

    return similarity_matrix


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

    del sim_dim_x, sim_dim_y, x

    return label_pair_matrix


def compute_pre_weight_matrix(labels_matrix, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method to compute the pre_weight_matrix used for the computation of the weight matrix
    :param labels_matrix: tensor matrix with the ground truth labels PixelX x PixelY
    :param dtype:
    :param device:
    :return: tensor PixelX x PixelY with weighted pixels for every label (1/number of pixels with the same label
    """
    # calculate pre-weighting matrix
    pre_weights_matrix = labels_matrix
    unique_labels = torch.unique(labels_matrix, sorted=True)
    for _, l in enumerate(unique_labels):
        pre_weights_matrix = torch.where(pre_weights_matrix == l, torch.div(torch.tensor(
            1, dtype=dtype, device=device), ((labels_matrix == l).nonzero()).size(0)), pre_weights_matrix)

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

    del sim_dim_x, sim_dim_y, x

    return weight_matrix


def get_embedding_loss(embedding_list, labels, dtype=c.data['dtype'], device=c.cuda['device']):
    """
    Method to compute the accumulated loss after the several MS iterations
    :param embedding_list: List of Embeddings Computed by the Net
    :param labels: Ground Truth of Labels
    :param dtype:
    :param device:
    :return: Tensor, returns the accumulated loss
    """
    loss = torch.tensor(0., dtype=dtype, device=device)
    for i in range(1, embedding_list.size(0)):
        loss = torch.add(loss, embedding_loss(embedding_matrix=embedding_list[i, :, :, :], labels=labels, dtype=dtype,
                                              device=device))
    return loss

# model_UNet = UNet()
# print(model_UNet)
#
# input = torch.randn(1, 1, 128, 128)
# out = model_UNet(input)
# print(out)
# print(out.size())
