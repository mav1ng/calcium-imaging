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
    """
    Implementation fo UNet Network
    Expects tensor input of dimension B x C x W x H, where B is Batch size, C is number of Channels, W width of image
    H height of image
    """

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

        self.Softmax2d = nn.Softmax2d()

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
        'should really use softmax here?'
        x = self.Softmax2d(x).clone()

        return x


class MS(nn.Module):

    def __init__(self):
        super(MS, self).__init__()

        self.emb = c.mean_shift['embedding_dim']
        # setting kernel bandwidth
        if c.mean_shift['kernel_bandwidth'] is not None:
            self.kernel_bandwidth = c.mean_shift['kernel_bandwidth']
        else:
            self.kernel_bandwidth = 1. / (1. - c.embedding_loss['margin']) / 3.
        self.step_size = c.mean_shift['step_size']
        self.iter = c.mean_shift['nb_iterations']
        self.bs = None
        self.w = None
        self.h = None

    # x_in flattened image in D x N , N number of Pixels

    def forward(self, x_in):
        """
        :param x_in: flattened image in D x N , D embedding Dimension, N number of Pixels
        :return: tensor with dimension B x I x D x W x H, B batch size, I number of iterations, D embedding dimension,
        W width of the image, H height of the image, embeddings x_in mean shifted
        """

        with torch.no_grad():
            self.bs = x_in.size(0)
            self.emb = x_in.size(1)
            self.w = x_in.size(2)
            self.h = x_in.size(3)

        x = x_in.view(self.bs, self.emb, -1)

        y = torch.zeros(self.iter + 1, self.emb, self.w * self.h).cuda()
        out = torch.zeros(self.bs, self.iter + 1, self.emb, self.w, self.h).cuda()

        # iterating over all samples in the batch
        for b in range(self.bs):
            y[0, :, :] = x[b, :, :]
            for t in range(1, self.iter + 1):
                kernel_mat = torch.exp(
                    torch.mul(self.kernel_bandwidth, mm(y[t - 1, :, :].clone().t(), y[t - 1, :, :].clone())))
                diag_mat = torch.diag(mm(kernel_mat.t(), torch.ones(self.w * self.h, 1).cuda())[:, 0], diagonal=0)

                y[t, :, :] = mm(y[t - 1, :, :].clone(),
                                torch.add(torch.mul(self.step_size, mm(kernel_mat, torch.inverse(diag_mat))),
                                          torch.mul(1. - self.step_size, torch.eye(self.w * self.h).cuda())))

            out[b, :, :, :, :] = y.view(self.iter + 1, self.emb, self.w, self.h)

        return out


class UNetMS(nn.Module):
    def __init__(self):
        super(UNetMS, self).__init__()

        self.UNet = UNet()
        self.MS = MS()
        self.L2Norm = L2Norm()

    def forward(self, x):
        x = self.UNet(x)
        x = self.L2Norm(x)
        x = self.MS(x)
        return x


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

        self.w = c.training['img_size']
        self.h = c.training['img_size']
        self.emb = c.UNet['embedding_dim']
        self.bs = c.training['batch_size']

    def forward(self, x):
        # L2 Normalization
        """Does the normalization really work that way?"""
        # bringing the embedding on the unit sphere
        # for b in range(x.size(0)):
        #     x[b] = F.normalize(x[b].clone().view(self.emb, -1), p=2, dim=0).view(self.emb, self.w, self.h)
        for b in range(x.size(0)):
            y = x[b].view(self.emb, -1)
            y_ = torch.mean(y, dim=0)
            y_n = y - y_
            y_n_ = torch.sqrt(torch.sum(y_n ** 2, dim=0))
            x[b] = (y_n / y_n_).view(self.emb, self.w, self.h)
        return x


class EmbeddingLoss(nn.Module):
    def __init__(self):
        super(EmbeddingLoss, self).__init__()

    def forward(self, emb, lab):
        """
        :param emb: Bs x Ch x w x h
        :param lab: bs x w x h
        :return: scalar loss
        """
        return embedding_loss(emb, lab)


def comp_similarity_matrix(input):
    """
    Method that computest the cosine similarity matrix
    input has dimensions Bs x Channels x Width x Height
    :param input:
    :return: N x N x 1 x Bs
    """
    (bs, ch, w, h) = input.size()
    out = torch.zeros((h * w, h * w, 1, bs)).cuda()

    for i in range(bs):
        sim = input[i].view(ch, w * h)

        sim_ = torch.mean(sim, dim=0)
        sim_n = sim - sim_
        sim__ = torch.sqrt(torch.sum(sim_n ** 2, dim=0))
        sim = (sim_n / sim__).t()

        # sim_ = torch.mean(sim, dim=0)
        # sim_n = sim - sim_
        # sim__ = torch.sqrt(torch.sum(sim_n ** 2, dim=0))
        # sim = (sim_n / sim__)

        out[:, :, 0, i] = torch.mm(sim, sim.t()) * 0.5 + 0.5

    return out


def compute_pre_weight_matrix(input):
    """
    Method to compute the pre_weight_matrix used for the computation of the weight matrix
    :param input: tensor matrix with the ground truth labels BatchSize x PixelX x PixelY
    :param dtype:
    :param device:
    :return: tensor BatchSize x PixelX x PixelY with weighted pixels for every label (1/number of pixels with the same label
    """

    (bs, w, h) = input.size()
    out = torch.zeros_like(input).cuda()

    for i in range(bs):
        with torch.no_grad():
            # calculate pre-weighting matrix
            out[i] = input[i]
            unique_labels = torch.unique(input[i], sorted=False)
            for _, l in enumerate(unique_labels):
                out[i] = torch.where(out[i] == l.float(), torch.div(torch.tensor(
                    1.).cuda(), torch.tensor(((input[i] == l).nonzero().size(0)))), out[i])
    return out


def compute_weight_matrix(input):
    """
    Method to compute the weight_matrix for the loss function
    :param input: tensor BatchSize x PixelX x PixelY with weighted pixels for every label (1/number of pixels with the
    same label
    :param dtype:
    :param device:
    :return: weight matrix N x N x 1 x Bs, where it is a triangular matrix with zeroes
    from the diagonal and below because this would be repeating information and in the upper part of the matrix the
    weight pairs for the computation of the loss function
    """

    (bs, w, h) = input.size()

    out = torch.zeros((h * w, h * w, 1, bs)).cuda()

    for i in range(bs):
        sim = input[i].view(h * w, 1)
        out[:, :, 0, i] = torch.mm(sim, sim.t())

    return out


def compute_label_pair(input):
    """
    Method that computes whether a pixel pair is of the same label or not
    :param input: tensor matrix with the ground truth labels Bs x PixelX x PixelY
    :param dtype:
    :param device:
    :return: tensor label pair matrix N x N x 1 x Bs, where it is a triangular matrix with zeroes
    from the diagonal and below because this would be repeating information and in the upper part of the matrix 1 if
    positive pair, -1 if negative pair, 0 if repeating information
    """

    (bs, w, h) = input.size()
    # +1 such that background has label != 0 such that following computation works
    y = torch.add(input.to(torch.float), 1.)

    out = torch.zeros((h * w, h * w, 1, bs)).cuda()

    # 1 if positive pair, -1 if negative pair, 0 if repeating information
    for i in range(bs):
        sim = y[i].view(h * w, 1)
        out[:, :, 0, i] = torch.mm(sim, sim.t())
        out[:, :, 0, i] = torch.where(torch.sqrt(out[:, :, 0, i]) == torch.mm(sim, torch.ones_like(sim.t())),
                                      torch.tensor(1.).cuda(), torch.tensor(-1.).cuda())

    return out


def embedding_loss(emb, lab):
    """
    Method that exhaustively calculates the embedding loss of an embedding when given the labels
    :param embedding_matrix: matrix with the predicted embeddings Bs x Iter + 1 x C x PixelX x PixelY
    :param labels: torch tensor with the ground truth Bs x w x h
    :param dtype: dtype of the tensors
    :param device: device which cuda uses
    :return: the total loss
    """

    (bs, iter, ch, w, h) = emb.size()

    loss = torch.zeros(bs, iter, w * h, w * h).cuda()
    weights = compute_weight_matrix(compute_pre_weight_matrix(lab))
    label_pairs = compute_label_pair(lab)

    for i in range(iter):
        sim_mat = comp_similarity_matrix(emb[:, i])
        for b in range(bs):
            loss[b, i] = torch.where(label_pairs[:, :, 0, b] == 1., torch.sub(1., sim_mat[:, :, 0, b]),
                                     loss[b, i])
            # correcting machine inaccuracies
            loss[b, i] = torch.where(loss[b, i] < 0., torch.tensor(0.).cuda(), loss[b, i])
            loss[b, i] = torch.where(label_pairs[:, :, 0, b] == -1.,
                                     torch.where(sim_mat[:, :, 0, b] - c.embedding_loss['margin'] >= 0,
                                                 sim_mat[:, :, 0, b] - c.embedding_loss['margin'], torch.tensor(
                                             0.).cuda()), loss[b, i])
            loss[b, i] = torch.mul(1. / (w * h), torch.sum(torch.mul(weights[:, :, 0, b], loss[b, i])))

    return torch.sum(loss) / iter


def scaling_loss(loss_vec, bs, nb_gpus):
    """
    method that scales the loss correctly when useing multiple gpus
    :param loss_vec: vector of the loss outputs
    :param bs: used batch size
    :param nb_gpus: number of used gpus
    :return: scaled scalar loss
    """
    assert bs >= nb_gpus, 'Batch Size should be bigger than the number of working gpus'
    out = 0.
    rem = bs % nb_gpus

    # weighing the single gpus regarding their batches
    if rem != 0:
        nb_gpus = nb_gpus - 1
        out = out + loss_vec[-1] / rem
    b = (bs - rem) / float(nb_gpus)
    for g in range(nb_gpus):
        out = out + loss_vec[g] / b

    # weighing the loss depending on the total number of gpus
    if rem != 0:
        nb_gpus = nb_gpus + 1

    return out / nb_gpus

# model_UNet = UNet()
# print(model_UNet)
#
# input = torch.randn(1, 1, 128, 128)
# out = model_UNet(input)
# print(out)
# print(out.size())
