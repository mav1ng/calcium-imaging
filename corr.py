import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import torch.nn.functional as F
import config as c


def get_big_star_mask():
    """
    :return: offsets of correlation pixels
    """
    indices_list = torch.tensor([(-1, 0), (0, -1), (1, 0), (0, 1), (0, -6), (5, -2), (5, 2), (0, 6), (-5, 2), (-5, -2),
                                 (0, -18), (15, -6), (15, 6), (0, 18), (-15, 6), (-15, -6)])
    return indices_list


def get_small_star_mask():
    """
    :return: offsets correlation pixels in flattened image
    """
    indices_list = torch.tensor([(-1, 0), (0, -1), (1, 0), (0, 1), (0, -6), (5, -2), (5, 2), (0, 6), (-5, 2), (-5, -2)])
    return indices_list


def get_corr(self, input_video, corr_form=c.corr['corr_form']):
    """
    With Tips of Elke: New Way of Calculating Corrs with two less for loops
    :param self:
    :param input_video: T * N * N,  N number of pixels on one side, T number of Frames
    :return:
    """

    X = input_video.size(1)
    Y = input_video.size(2)

    corr_mask = torch.tensor([])

    # choosing correlation form here
    if corr_form == 'big_star':
        corr_mask = get_big_star_mask()

    if corr_form == 'small_star':
        corr_mask = get_small_star_mask()

    correlation_pic = torch.zeros((corr_mask.size(0), X, Y))

    u = input_video
    u_ = torch.mean(u, dim=0)
    u_u_ = u - u_
    u_u_n = torch.sqrt(torch.sum(u_u_ ** 2, dim=0))

    for o, off in enumerate(corr_mask):
        print(o)
        v = torch.zeros(input_video.size())
        v[:, off[0]:, off[1]:] = input_video[:, :(input_video.size(1) - off[0]), :(input_video.size(2) - off[1])]
        v_ = torch.mean(v, dim=0)
        v_v_ = v - v_
        v_v_n = torch.sqrt(torch.sum(v_v_ ** 2, dim=0))

        zaehler = torch.sum(torch.mul(u_u_, v_v_), dim=0)
        nenner = torch.mul(u_u_n, v_v_n)

        correlation_pic[o] = torch.where(nenner > 0., zaehler.div(nenner), torch.zeros((X, Y)))

    return correlation_pic


def get_sliced_corr(input_video, corr_form=c.corr['corr_form'], slice_size=100):
    """
    Method that computes the correlations for a series of images with the slicing process
    :param input_video: T * N * N,  N number of pixels on one side, T number of Frames
    :param corr_form: form of the correlation mask that is used for calculating the correlations
    :param slice_size: number of frames in one slice
    :return: C * N * N, C number of correlations
    """
    corr_array = torch.tensor([])
    startidx = np.random.randint(0, input_video.size()[0] - 1)
    number_iter = int(np.floor(input_video.size()[0] / slice_size))

    rolled_input_video = torch.roll(input_video, shifts=(startidx, 0, 0), dims=(0, 1, 2))

    for i in range(number_iter):
        if i == 0:
            corr_array = torch.reshape(get_corr(rolled_input_video[i * slice_size: (i + 1) * slice_size, :, :],
                                                 corr_form), (-1, input_video.size(1),
                                                              input_video.size(2), 1))
        else:
            corr_array = torch.cat(
                (corr_array, torch.reshape(
                    get_corr(
                        rolled_input_video[i * slice_size: (i + 1) * slice_size, :, :], corr_form
                    ), (-1, input_video.size()[1], input_video.size()[2], 1))), dim=3)

    return torch.max(corr_array, dim=3)[0]
