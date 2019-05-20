import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr


def get_big_star_mask():
    """
    :return: offsets of correlation pixels
    """
    indices_list = np.array([(-1, 0), (0, -1), (1, 0), (0, 1), (0, -6), (5, -2), (5, 2), (0, 6), (-5, 2), (-5, -2),
                             (0, -18), (15, -6), (15, 6), (0, 18), (-15, 6), (-15, -6)])
    return indices_list


def get_small_star_mask():
    """
    :return: offsets correlation pixels in flattened image
    """
    indices_list = np.array([(-1, 0), (0, -1), (1, 0), (0, 1), (0, -6), (5, -2), (5, 2), (0, 6), (-5, 2), (-5, -2)])
    return indices_list


def calc_corr(input_video, input_line, corr_mask, coordinates):
    """
    Calculates Correlation for One Pixel, specifying correlation mask
    :param input_video:
    :param input_line:
    :param corr_mask:
    :param coordinates:
    :return:
    """

    '''working here right now trying to fix'''
    corr_mask = corr_mask + coordinates
    correlations = np.zeros(corr_mask.shape[0])

    for idx, corr_pixel in enumerate(corr_mask):
        if (corr_pixel <= -1).any() or (corr_pixel >= input_video.shape[1]).any():
            correlations[idx] = 0
        else:
            #print(corr_pixel[0], corr_pixel[1])
            correlations[idx] = pearsonr(input_line, input_video[:, corr_pixel[0], corr_pixel[1]])[0]
    return correlations


def get_corr(input_video, corr_form):
    """
    :param input_video: T * N * N,  N number of pixels on one side, T number of Frames
    needs to be normalized
    :return: C * N * N, C number of correlations
    """

    pixel_nb = input_video.shape[1]

    corr_mask = torch.Tensor()

    # choosing correlation form here
    if corr_form == 'big_star':
        corr_mask = get_big_star_mask()

    if corr_form == 'small_star':
        corr_mask = get_small_star_mask()

    correlation_pic = np.zeros((corr_mask.shape[0], pixel_nb, pixel_nb))

    for x in range(input_video.shape[1] - 1):
        for y in range(input_video.shape[2] - 1):
            correlation_pic[:, x, y] = calc_corr(input_video, input_video[:, x, y], corr_mask, (x, y))

    return correlation_pic


def get_sliced_corr(input_video, corr_form, slice_size=100):
    corr_array = np.array([])
    startidx = np.random.randint(0, input_video.size()[0] - 1)
    number_iter = int(np.floor(input_video.size()[0] / slice_size))

    rolled_input_video = np.roll(input_video, startidx, axis=0)
    print(str(rolled_input_video.shape) + 'herere 1')

    for i in range(number_iter):
        if i == 0:
            corr_array = np.reshape(get_corr(rolled_input_video[i * slice_size: (i + 1) * slice_size, :, :],
                                                               corr_form), (-1, input_video.size()[1],
                                                                            input_video.size()[2], 1))
        else:
            corr_array = np.concatenate(
                (corr_array, np.reshape(
                    get_corr(
                        rolled_input_video[i * slice_size: (i + 1) * slice_size, :, :], corr_form
                    ), (-1, input_video.size()[1], input_video.size()[2], 1))), axis=3)

    print(corr_array.shape)
    return np.amax(corr_array, axis=3)


a = torch.randn((30, 50, 50))
print(a.size())
b = get_sliced_corr(a, corr_form='small_star', slice_size=5)
print(b.shape)







