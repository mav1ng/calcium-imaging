import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import torch.nn.functional as F
import config as c


def calc_corr(input_video, input_line, corr_mask, coordinates):
    """
    Calculates Correlation for One Pixel, specifying correlation mask
    :param input_video: torch tensor
    :param input_line: torch tensor
    :param corr_mask:
    :param coordinates:
    :return:
    """

    '''working here right now trying to fix'''
    corr_mask = corr_mask + torch.tensor(coordinates)
    correlations = torch.zeros(corr_mask.shape[0])

    for idx, corr_pixel in enumerate(corr_mask):
        if (corr_pixel <= -1).any() or (corr_pixel >= input_video.size()[1]).any():
            correlations[idx] = 0
        else:
            # print(corr_pixel[0], corr_pixel[1])
            correlations[idx] = torch.tensor(pearsonr(input_line, input_video[:, corr_pixel[0], corr_pixel[1]])[0])
    return correlations


def get_corr(input_video, corr_form=c.corr['corr_form']):
    """
    Method that computes the correlations for a series of images
    :param corr_form: form of the correlation mask that is used for calculating the correlations
    :param input_video: T * N * N,  N number of pixels on one side, T number of Frames
    needs to be normalized
    :return: C * N * N, C number of correlations
    """

    pixel_nb = input_video.size(1)

    corr_mask = torch.tensor([])

    # choosing correlation form here
    if corr_form == 'big_star':
        corr_mask = get_big_star_mask()

    if corr_form == 'small_star':
        corr_mask = get_small_star_mask()

    correlation_pic = torch.zeros((corr_mask.size(0), pixel_nb, pixel_nb))

    for x in range(input_video.size(1)):
        print(x)
        for y in range(input_video.size(2)):
            correlation_pic[:, x, y] = calc_corr(input_video, input_video[:, x, y], corr_mask, (x, y))

    return correlation_pic