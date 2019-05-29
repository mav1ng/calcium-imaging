import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul as mm

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
    return 1 / 2 * (1 + mm(vec1.t(), vec2) / (torch.norm(vec1) * torch.norm(vec2)))


def get_embedding_loss(list_embeddings):
    pass


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
        self.embeddings_list = []
        self.nb_pixels = None  # to be defined when forward is called
        self.pic_res = None
        self.device = c.cuda['device']

    # x_in flattened image in D x N , N number of Pixels

    def forward(self, x_in):
        """
        :param x_in: flattened image in D x N , N number of Pixels
        :return: embeddings x_in mean shifted
        """
        self.pic_res = x_in.size(3)
        x = x_in.view(self.embedding_dim, -1)
        self.nb_pixels = x.size(1)
        print('input Mean Shift Block' + str(x.size()))
        for t in range(self.nb_iterations):
            print('Mean Shift: ' + str(t) + ' First Iteration')
            # kernel_mat N x N , N number of pixels
            kernel_mat = torch.exp(torch.mul(self.kernel_bandwidth, mm(x.t(), x)))
            # diag_mat N x N
            diag_mat = torch.diag(
                mm(kernel_mat.t(), torch.ones((self.nb_pixels, 1), device=self.device)).squeeze(dim=1), diagonal=0)

            x = mm(x,
                   torch.mul(self.step_size, mm(kernel_mat, torch.inverse(diag_mat))) +
                   torch.mul((1 - self.step_size), torch.eye(self.nb_pixels, self.nb_pixels, device=self.device)))
            '''
            PERHAPS CALCULATE LOSS BEFORE PUTTING INTO LIST BECAUSE OF MEMORY PROBLEMS'''
            self.embeddings_list.append(x.view(self.embedding_dim, self.pic_res, self.pic_res))
        return x.view(self.embedding_dim, self.pic_res, self.pic_res), self.embeddings_list


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
    pre_weights_matrix = torch.ones_like(labels, dtype=dtype, device=device)
    weights_matrix = torch.zeros_like(labels_matrix)

    for w in range(labels.size(0)):
        for v in range(labels.size(1)):
            pre_weights_matrix[w, v] = torch.div(torch.tensor(1, dtype=dtype, device=device),
                                                 ((labels == labels[w, v]).nonzero()).size(0))

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

    indices_positive = (labels_matrix == 1).nonzero()
    indices_negative = (labels_matrix == -1).nonzero()

    loss = torch.zeros(pic_dim[0] * pic_dim[1], pic_dim[0] * pic_dim[1], dtype=dtype, device=device)
    loss[indices_positive] = torch.sub(1, similarity_matrix[indices_positive])
    loss[indices_negative] = torch.where(similarity_matrix[indices_negative] - c.embedding_loss['margin'] >= 0,
                                         similarity_matrix[indices_negative] - c.embedding_loss['margin'], torch.tensor(
            0, device=device, dtype=dtype))

    loss = torch.mul(1 / (pic_dim[0] * pic_dim[1]), torch.sum(torch.mul(weights_matrix, loss)))

    return loss

# model_UNet = UNet()
# print(model_UNet)
#
# input = torch.randn(1, 1, 128, 128)
# out = model_UNet(input)
# print(out)
# print(out.size())
