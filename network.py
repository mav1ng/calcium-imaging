import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.matmul as mm

import config


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
    return 1/2 * (1 + mm(vec1.t(), vec2) / (torch.norm(vec1) * torch.norm(vec2)))


def get_embedding_loss(list_embeddings):
    pass


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.input_channels = config.UNet['input_channels']
        self.embedding_dim = config.UNet['embedding_dim']
        self.dropout_rate = config.UNet['dropout_rate']

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
        self.conv_layer_end = nn.Conv2d(self.embedding_dim, 2, 1)

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = F.relu(x)
        x = self.conv_layer_2(x)
        x = F.relu(x)
        conc_in_1 = x   # used later when filters are concatenated

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

        self.embedding_dim = config.mean_shift['embedding_dim']
        # setting kernel bandwidth
        if config.mean_shift['kernel_bandwidth'] is not None:
            self.kernel_bandwidth = config.mean_shift['kernel_bandwidth']
        else:
            self.kernel_bandwidth = 1 / (1 - config.embedding_loss['margin'])/3
        self.step_size = config.mean_shift['step_size']
        self.nb_iterations = config.mean_shift['nb_iterations']
        self.embeddings_list = []
        self.nb_pixels = None  # to be defined when foward is called

    # x_in flattened image in D x N , N number of Pixels

    def forward(self, x_in):
        """
        :param x_in: flattened image in D x N , N number of Pixels
        :return: embeddings x_in mean shifted
        """
        x = x_in
        self.nb_pixels = x_in.size()[1]
        for t in range(self.nb_iterations):
            # kernel_mat N x N , N number of pixels
            kernel_mat = torch.exp(mm(self.kernel_bandwidth, mm(x.t(), x)))
            # diag_mat N x N
            diag_mat = torch.diag(mm(kernel_mat.t(), torch.ones(1, self.nb_pixels)), diagonal=0)
            x = mm(x,
                   mm(self.step_size, mm(kernel_mat, torch.inverse(diag_mat))) +
                   mm((1 - self.step_size), torch.eye(self.nb_pixels, self.nb_pixels)))
            self.embeddings_list.append(x)
        return x


class UNet_MS(nn.Module):
    def __init__(self):
        super(UNet_MS, self).__init__()

        self.UNet = UNet()
        self.MS = MS()

    def forward(self, x):
        x = self.UNet(x)
        """Does the normalization really work that way?"""
        x = F.normalize(x, p=2, dim=1)
        x = self.MS(x)
        return x


model_UNet = UNet()
print(model_UNet)

input = torch.randn(1, 1, 128, 128)
out = model_UNet(input)
print(out)
print(out.size())
