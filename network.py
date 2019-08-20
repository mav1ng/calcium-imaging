import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import matmul as mm
import numpy as np
import matplotlib.pyplot as plt

import config as c
import helpers as h
import helpers as he


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

    def __init__(self, input_channels=c.UNet['input_channels'], embedding_dim=c.UNet['embedding_dim'],
                 dropout_rate=c.UNet['dropout_rate'], background_pred=False):
        super(UNet, self).__init__()

        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.background_pred = background_pred

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
        if self.background_pred:
            self.conv_layer_end = nn.Conv2d(self.embedding_dim, self.embedding_dim + 2, 1)
        else:
            self.conv_layer_end = nn.Conv2d(self.embedding_dim, self.embedding_dim, 1)

        self.Softmax2d = nn.Softmax2d()

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = F.relu(x)
        x = self.conv_layer_2(x)
        x = F.relu(x)
        conc_in_1 = x.clone()  # used later when filters are concatenated

        x = self.max_pool_2D_1(x)
        x = self.conv_layer_3(x)
        x = F.relu(x)
        x = self.conv_layer_4(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        conc_in_2 = x.clone()

        x = self.max_pool_2D_2(x)
        x = self.conv_layer_5(x)
        x = F.relu(x)
        x = self.conv_layer_6(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        conc_in_3 = x.clone()

        x = self.max_pool_2D_3(x)
        x = self.conv_layer_7(x)
        x = F.relu(x)
        x = self.conv_layer_8(x)
        x = F.relu(x)
        x = self.dropout_3(x)
        conc_in_4 = x.clone()

        x = self.max_pool_2D_4(x)
        x = self.conv_layer_9(x)
        x = F.relu(x)
        x = self.conv_layer_10(x)
        x = F.relu(x)
        x = self.up_layer_1(x)
        x = F.relu(x)
        x = self.dropout_4(x)

        x = torch.cat((x.clone(), conc_in_4), dim=1)
        x = self.conv_layer_11(x)
        x = F.relu(x)
        x = self.conv_layer_12(x)
        x = F.relu(x)
        x = self.up_layer_2(x)
        x = F.relu(x)
        x = self.dropout_5(x)

        x = torch.cat((x.clone(), conc_in_3), dim=1)
        x = self.conv_layer_13(x)
        x = F.relu(x)
        x = self.conv_layer_14(x)
        x = F.relu(x)
        x = self.up_layer_3(x)
        x = F.relu(x)
        x = self.dropout_6(x)

        x = torch.cat((x.clone(), conc_in_2), dim=1)
        x = self.conv_layer_15(x)
        x = F.relu(x)
        x = self.conv_layer_16(x)
        x = F.relu(x)
        x = self.up_layer_4(x)
        x = F.relu(x)
        x = self.dropout_7(x)

        x = torch.cat((x.clone(), conc_in_1), dim=1)
        x = self.conv_layer_17(x)
        x = F.relu(x)
        x = self.conv_layer_18(x)
        x = F.relu(x)
        x = self.conv_layer_end(x)
        'should really use softmax here?'
        x = self.Softmax2d(x).clone()

        return x


class MS(nn.Module):

    def __init__(self, embedding_dim=c.mean_shift['embedding_dim'], kernel_bandwidth=c.mean_shift['kernel_bandwidth'],
                 margin=c.embedding_loss['margin'], step_size=c.mean_shift['step_size'],
                 nb_iterations=c.mean_shift['nb_iterations'], use_in_val=c.mean_shift['use_in_val'],
                 use_embedding_loss=c.embedding_loss['on'], scaling=c.embedding_loss['scaling'],
                 use_background_pred=c.UNet['background_pred'],
                 include_background=c.embedding_loss['include_background']):
        super(MS, self).__init__()

        self.emb = embedding_dim
        self.margin = margin
        self.scaling = scaling
        self.include_background = include_background

        self.kernel_bandwidth = kernel_bandwidth
        if self.kernel_bandwidth is None:
            self.kernel_bandwidth = 3. / (1. - self.margin)
        self.step_size = step_size
        self.nb_iterations = nb_iterations
        self.iter = nb_iterations

        self.use_in_val = use_in_val
        self.use_embedding_loss = use_embedding_loss
        self.use_background_pred = use_background_pred

        self.bs = None
        self.w = None
        self.h = None
        self.val = False
        self.test = False

        self.prefer_cell = 0.5

        self.criterion = EmbeddingLoss(margin=self.margin).cuda()

    def forward(self, x_in, lab_in, subsample_size, background_pred=None):
        """
        :param x_in: flattened image in D x N , D embedding Dimension, N number of Pixels
        :param lab_in specify labels B x W x H if model is usd in training mode
        :return: tensor with dimension B x D x W x H, B batch size, D embedding dimension, loss
        W width of the image, H height of the image, embeddings x_in mean shifted
        """

        d = torch.device('cuda:0')

        (self.bs, self.emb, self.w, self.h) = x_in.size()
        (self.sw, self.sh) = (self.w, self.h)

        # with torch.no_grad():
        #     self.bs = x_in.size(0)
        #     self.emb = x_in.size(1)
        #     self.w = x_in.size(2)
        #     self.h = x_in.size(3)

        x = x_in.view(self.bs, self.emb, self.w, self.h)
        out = torch.zeros(self.bs, self.emb, self.w, self.h, device=d)
        out = x_in.view(self.bs, self.emb, self.w, self.h)
        y = torch.zeros(self.emb, self.w * self.h, device=d)

        if self.val and not self.test:
            """Validation SUBSAMPLING"""
            val_sub_size = 100 * 100
            emb, lab, ind = he.emb_subsample(x.clone(), lab_in.clone(), include_background=True,
                                             backpred=None, prefer_cell=0.5,
                                             sub_size=val_sub_size)

            x = emb.view(self.bs, self.emb, -1)
            y = torch.zeros(self.emb, val_sub_size, device=d)
            wurzel = int(np.sqrt(val_sub_size))
            self.sw = wurzel
            self.sh = wurzel
        elif subsample_size is not None and not self.test:
            """Training SUBSAMPLING"""
            emb, lab, ind = he.emb_subsample(x.clone(), lab_in.clone(), include_background=self.include_background,
                                             backpred=background_pred, prefer_cell=self.prefer_cell,
                                             sub_size=subsample_size)
            x = emb.view(self.bs, self.emb, -1)
            y = torch.zeros(self.emb, subsample_size, device=d)
            wurzel = int(np.sqrt(subsample_size))
            self.sw = wurzel
            self.sh = wurzel

        ret_loss = 0.

        if (not self.use_in_val) and (not self.training or self.val):
            self.iter = 0
        elif self.iter == 0:
            self.iter = self.nb_iterations

        for t in range(self.iter + 1):
            # iterating over all samples in the batch
            for b in range(self.bs):
                y = x[b, :, :]
                if t != 0:
                    kernel_mat = torch.exp(torch.mul(self.kernel_bandwidth, mm(y.clone().t(), y.clone())))
                    diag_mat = torch.diag(mm(kernel_mat.t(), torch.ones(self.sw * self.sh, 1, device=d))[:, 0],
                                          diagonal=0)
                    y = mm(y.clone(),
                           torch.add(torch.mul(self.step_size, mm(kernel_mat, torch.inverse(diag_mat))),
                                     torch.mul(1. - self.step_size, torch.eye(self.sw * self.sh, device=d))))

                if subsample_size is not None and not self.test:
                    out = out.view(self.bs, self.emb, -1)
                    out[b, :, ind] = y
                    out = out.view(self.bs, self.emb, self.w, self.h)
                else:
                    out[b, :, :, :] = y.view(self.emb, self.w, self.h)

            x = out.view(self.bs, self.emb, -1)
            if subsample_size is not None and not self.test:
                x = out.view(self.bs, self.emb, -1)[:, :, ind]

            # print('self.training', self.training)

            if self.training and self.use_embedding_loss and not self.val:
                lab_in_ = torch.tensor(h.get_diff_labels(lab_in.detach().cpu().numpy()), device=d)

                emb = out
                lab = lab_in_

                if subsample_size is not None:
                    emb = out.view(self.bs, self.emb, -1)[:, :, ind].view(self.bs, self.emb, wurzel, wurzel)
                    lab = lab_in_.view(self.bs, -1)[:, ind].view(self.bs, wurzel, wurzel)

                loss = self.criterion(emb, lab)

                loss = (loss / self.bs) * self.scaling * (1/(self.iter + 1))

                with torch.no_grad():
                    ret_loss = ret_loss + loss.detach()

                if t == self.iter and not self.use_background_pred and not t == 0:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
            elif self.val and not self.test:
                lab_in_ = torch.tensor(h.get_diff_labels(lab_in.detach().cpu().numpy()), device=d)

                emb = out
                lab = lab_in_

                if subsample_size is not None:
                    emb = out.view(self.bs, self.emb, -1)[:, :, ind].view(self.bs, self.emb, wurzel, wurzel)
                    lab = lab_in_.view(self.bs, -1)[:, ind].view(self.bs, wurzel, wurzel)

                val_criterion = EmbeddingLoss(margin=0.5).cuda()
                loss = val_criterion(emb, lab)

                loss = (loss / self.bs) * 25 * (1/(self.iter + 1))

                with torch.no_grad():
                    ret_loss = ret_loss + loss.detach()

        out = x_in

        return out, ret_loss


class UNetMS(nn.Module):
    def __init__(self, input_channels=c.UNet['input_channels'], embedding_dim=c.UNet['embedding_dim'],
                 dropout_rate=c.UNet['dropout_rate'], kernel_bandwidth=c.mean_shift['kernel_bandwidth'],
                 margin=c.embedding_loss['margin'], step_size=c.mean_shift['step_size'],
                 nb_iterations=c.mean_shift['nb_iterations'], use_in_val=c.mean_shift['use_in_val'],
                 use_embedding_loss=c.embedding_loss['on'], scaling=c.embedding_loss['scaling'],
                 use_background_pred=c.UNet['background_pred'], subsample_size=c.embedding_loss['subsample_size'],
                 include_background=c.embedding_loss['include_background']):
        super(UNetMS, self).__init__()

        # instantializing class parameters
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels

        self.margin = margin
        self.include_background = include_background

        self.step_size = step_size
        self.kernel_bandwidth = kernel_bandwidth
        self.nb_iterations = nb_iterations

        self.subsample_size = subsample_size

        self.use_in_val = use_in_val
        self.use_embedding_loss = use_embedding_loss
        self.scaling = scaling
        self.use_background_pred = use_background_pred

        # instantializing class models
        self.UNet = UNet(background_pred=self.use_background_pred, input_channels=self.input_channels,
                         embedding_dim=self.embedding_dim, dropout_rate=self.dropout_rate)
        self.MS = MS(embedding_dim=self.embedding_dim, margin=self.margin, step_size=self.step_size,
                     kernel_bandwidth=self.kernel_bandwidth, nb_iterations=self.nb_iterations,
                     include_background=self.include_background, use_in_val=self.use_in_val,
                     use_embedding_loss=self.use_embedding_loss,
                     scaling=self.scaling, use_background_pred=self.use_background_pred)
        self.L2Norm = L2Norm()

    def forward(self, x, lab):
        x = self.UNet(x)
        x = self.L2Norm(x)
        if self.use_background_pred:
            x = x.clone()[:, :-2]
            y = x[:, -2:]
            x, ret_loss = self.MS(x, lab, background_pred=y[:, 0], subsample_size=self.subsample_size)
            return x, ret_loss, y
        else:
            x, ret_loss = self.MS(x, lab, subsample_size=self.subsample_size)
            return x, ret_loss


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, x):
        # L2 Normalization
        (bs, c, w, h) = x.size()
        for b in range(bs):
            y = x[b].view(c, -1)
            y_ = torch.mean(y, dim=0)
            y_n = y - y_
            y_n_ = torch.sqrt(torch.sum(y_n ** 2, dim=0))
            x[b] = (y_n / y_n_).view(c, w, h)
        return x


class EmbeddingLoss(nn.Module):
    def __init__(self, margin):
        super(EmbeddingLoss, self).__init__()

        self.margin = margin

    def forward(self, emb, lab):
        """
        :param emb: Bs x Ch x w x h
        :param lab: bs x w x h
        :return: scalar loss
        """
        return embedding_loss(emb, lab, margin=self.margin)


def comp_similarity_matrix(input):
    """
    Method that computest the cosine similarity matrix
    input has dimensions Bs x Channels x Width x Height
    :param input:
    :return: N x N x 1 x Bs
    """

    d = torch.device('cuda:0')

    (bs, ch, w, h) = input.size()
    out = torch.zeros((h * w, h * w, 1, bs), device=d)

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

    d = torch.device('cuda:0')

    (bs, w, h) = input.size()
    out = torch.zeros_like(input, device=d)

    for i in range(bs):
        with torch.no_grad():
            # calculate pre-weighting matrix
            out[i] = input[i]
            unique_labels = torch.unique(input[i], sorted=False)
            for _, l in enumerate(unique_labels):
                out[i] = torch.where(out[i] == l.float(), torch.div(torch.tensor(
                    1., device=d), torch.tensor(((input[i] == l).nonzero().size(0)), device=d)), out[i])
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

    d = torch.device('cuda:0')

    (bs, w, h) = input.size()

    out = torch.zeros((h * w, h * w, 1, bs), device=d)

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

    d = torch.device('cuda:0')

    (bs, w, h) = input.size()

    # +1 such that background has label != 0 such that following computation works
    y = torch.add(input.to(torch.float), 1.)

    out = torch.zeros((h * w, h * w, 1, bs), device=d)

    # 1 if positive pair, -1 if negative pair, 0 if repeating information
    for i in range(bs):
        sim = y[i].view(h * w, 1)
        out[:, :, 0, i] = torch.mm(sim, sim.t())
        out[:, :, 0, i] = torch.where(torch.sqrt(out[:, :, 0, i]) == torch.mm(sim, torch.ones_like(sim.t())),
                                      torch.tensor(1., device=d), torch.tensor(-1., device=d))
        out[:, :, 0, i] = torch.where(torch.mm(sim, sim.t()) == 0., torch.tensor(0., device=d), out[:, :, 0, i])
    return out


def embedding_loss(emb, lab, margin):
    """
    Method that exhaustively calculates the embedding loss of an embedding when given the labels
    :param embedding_matrix: matrix with the predicted embeddings Bs x Iter + 1 x C x PixelX x PixelY
    :param labels: torch tensor with the ground truth Bs x w x h
    :param use_subsampling: Boolean wether to use subsampling or not
    :param dtype: dtype of the tensors
    :param device: device which cuda uses
    :return: the total loss
    """

    d = torch.device('cuda:0')

    (bs, ch, w, h) = emb.size()

    loss = torch.zeros(bs, w * h, w * h, device=d)
    weights = compute_weight_matrix(compute_pre_weight_matrix(lab))
    label_pairs = compute_label_pair(lab)
    sim_mat = comp_similarity_matrix(emb)

    for b in range(bs):
        loss[b] = torch.where(label_pairs[:, :, 0, b] == 1., torch.sub(1., sim_mat[:, :, 0, b]),
                              loss[b])
        # correcting machine inaccuracies
        loss[b] = torch.where(loss[b] < 0., torch.tensor(0., device=d), loss[b])
        loss[b] = torch.where(label_pairs[:, :, 0, b] == -1.,
                              torch.where(sim_mat[:, :, 0, b] - margin >= 0,
                                          sim_mat[:, :, 0, b] - margin, torch.tensor(
                                      0., device=d)), loss[b])

        loss[b] = torch.mul(1. / (w * h), torch.mul(weights[:, :, 0, b], loss[b].clone()))

    return torch.sum(loss)


def scaling_loss(loss_vec, bs, nb_gpus):
    """
    method that scales the loss correctly when useing multiple gpus
    :param loss_vec: vector of the loss outputs
    :param bs: used batch size
    :param nb_gpus: number of used gpus
    :return: scaled scalar loss
    """
    print('number of gpus', nb_gpus)
    print('batch size', bs)
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
