import numpy as np

def get_corr(input_video, corr_form):
    """
    :param input_video: T * N * N , video flattened, N number of pixels on one side, T number of Frames
    :return: C * N * N, video flattened, C number of correlations
    """

    idx_list = []
    if corr_form == 'big_star':
        corr_mask = get_big_star_mask()


def get_big_star_mask(idx):
    """
    :param idx: index of target pixel in video input
    :param window_size: size of image
    :return: indices of correlation pixels in flattened image
    """
    indices_list = [(-1, 0), (0, -1), (1, 0), (0, 1), (0, -6), (5, -2), (5, 2), (0, 6), (-5, 2), (-5, -2), (0, -18),
                    (15, -6), (15, 6), (0, 18), (-15, 6), (-15, -6)]
    ret_list = []
    for entry in indices_list:
        ret_list.append(idx + entry)
    return ret_list

def get_small_star_mask(idx):
    """
    :param idx: index of target pixel in video input
    :param window_size: size of image
    :return: indices of correlation pixels in flattened image
    """
    indices_list = [(-1, 0), (0, -1), (1, 0), (0, 1), (0, -6), (5, -2), (5, 2), (0, 6), (-5, 2), (-5, -2)]
    ret_list = []
    for entry in indices_list:
        ret_list.append(idx + entry)
    return ret_list
