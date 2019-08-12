import data
import helpers as h

from os import listdir
from os.path import isfile, join

import json
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from pprint import pprint


class Analysis:
    def __init__(self, model_name, input_channels,
                 embedding_dim,
                 background_pred,
                 mean_shift_on,
                 nb_iterations,
                 kernel_bandwidth,
                 step_size,
                 embedding_loss,
                 margin,
                 include_background,
                 scaling,
                 subsample_size,
                 learning_rate,
                 nb_epochs,
                 batch_size,
                 pre_train,
                 pre_train_name,
                 val_score=0.,
                 emb_score=-1.,
                 cel_score=-1.):

        self.epoch = 0

        self.model_name=model_name
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.background_pred = background_pred
        self.mean_shift_on = mean_shift_on
        self.nb_iterations = nb_iterations
        self.kernel_bandwidth = kernel_bandwidth
        self.step_size = step_size
        self.embedding_loss = embedding_loss
        self.margin = margin
        self.include_background = include_background
        self.scaling = scaling
        self.subsample_size = subsample_size
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.pre_train = pre_train
        self.pre_train_name = pre_train_name

        self.val_score = val_score
        self.emb_score = emb_score
        self.cel_score = cel_score


def create_analysis(json_file):

    dic = data.read_from_json(json_file)

    try:
        ana = Analysis(model_name=dic['model_name'],
                       input_channels=dic['input_channels'],
                       embedding_dim=dic['embedding_dim'],
                       background_pred=dic['background_pred'],
                       mean_shift_on=dic['Mean Shift On'],
                       nb_iterations=dic['nb_iterations'],
                       kernel_bandwidth=dic['kernel_bandwidth'],
                       step_size=dic['step_size'],
                       embedding_loss=dic['Embedding Loss'],
                       margin=dic['margin'],
                       include_background=dic['Include Background'],
                       scaling=dic['scaling'],
                       subsample_size=dic['subsample_size'],
                       learning_rate=dic['Learning Rate'],
                       nb_epochs=dic['nb_epochs'],
                       batch_size=dic['batch_size'],
                       pre_train=dic['pre_train'],
                       pre_train_name=dic['pre_train_name'],
                       val_score=dic['val_score'],
                       emb_score=dic['emb_score'],
                       cel_score=dic['cel_score'])
    except KeyError:
        print('Not yet Scored')
        ana = Analysis(model_name=dic['model_name'],
                       input_channels=dic['input_channels'],
                       embedding_dim=dic['embedding_dim'],
                       background_pred=dic['background_pred'],
                       mean_shift_on=dic['Mean Shift On'],
                       nb_iterations=dic['nb_iterations'],
                       kernel_bandwidth=dic['kernel_bandwidth'],
                       step_size=dic['step_size'],
                       embedding_loss=dic['Embedding Loss'],
                       margin=dic['margin'],
                       include_background=dic['Include Background'],
                       scaling=dic['scaling'],
                       subsample_size=dic['subsample_size'],
                       learning_rate=dic['Learning Rate'],
                       nb_epochs=dic['nb_epochs'],
                       batch_size=dic['batch_size'],
                       pre_train=dic['pre_train'],
                       pre_train_name=dic['pre_train_name'])

    return ana


def get_analysis(analysis_name):
    files = [f for f in listdir('config/') if isfile(join('config/', f)) and str(analysis_name) in f]
    ana_list = []
    for file in files:
        print('config/' + str(file))
        ana_list.append(create_analysis('config/' + str(file)))
    return ana_list


def val_score_analysis(analysis_list):
    ret = {}
    for ana in analysis_list:
        print('Evaluating ' + str(ana.model_name))
        (val, emb, cel) = h.val_score(model_name=str(ana.model_name), iter=10, th=0.8,
                                    background_pred=ana.background_pred, use_metric=False)
        ret[str(ana.model_name)] = (val, emb, cel)
    return ret


def score(analysis_name):
    ana_list = get_analysis(analysis_name)
    score_list = val_score_analysis(ana_list)

    for ana in ana_list:
        ana.val_score = score_list[ana.model_name][0]
        ana.emb_score = score_list[ana.model_name][1]
        ana.cel_score = score_list[ana.model_name][2]
        data.save_config_score(ana.model_name, ana.val_score, ana.emb_score, ana.cel_score, ana.input_channels,
                               ana.embedding_dim, ana.background_pred, ana.mean_shift_on, ana.nb_iterations,
                               ana.kernel_bandwidth, ana.step_size, ana.embedding_loss, ana.margin,
                               ana.include_background, ana.scaling, ana.subsample_size, ana.learning_rate,
                               ana.nb_epochs, ana.batch_size, ana.pre_train, ana.pre_train_name)
    pass


def analyse_lr_nbep_bs(analysis_name):
    ana_list = get_analysis(analysis_name)

    ana_param = ana_list[0]

    ret = np.zeros((6, ana_list.__len__()))

    for i, ana in enumerate(ana_list):
        ret[0, i] = float(ana.learning_rate)
        ret[1, i] = float(ana.nb_epochs)
        ret[2, i] = float(ana.batch_size)
        ret[3, i] = float(ana.val_score)
        ret[4, i] = float(ana.emb_score)
        ret[5, i] = float(ana.cel_score)

    optimized_parameters = ['Learning Rate', 'Number of Epochs', 'Batch Size']

    return ret, ana_param, optimized_parameters


def normalize_score(input):
    """
    Method that normalizes the 1D input array
    :param input: 1D Array
    :return: normalized 1D array
    """

    d_ = np.nanmean(input)
    d_n = input - d_
    d_n_ = np.sqrt(np.nansum(d_n ** 2))
    ret = d_n / d_n_

    return ret


def plot_lr_nbep_bs(analysis_name):
    """
    Method that visualizes the Perfomance of a model depending on Learning Rate, Number of Epochs and Batch size
    in a 4D graph by normalizing and weighting embedding and cel score perhaps not smart
    :param analysis_name:
    :return:
    """
    data, ana_param, _ = analyse_lr_nbep_bs(analysis_name)

    data[4] = normalize_score(data[4])
    data[5] = normalize_score(data[5])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data[0]
    y = data[1]
    z = data[2]
    c = ((data[4] + data[5]) / 2)

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)
    plt.show()


def find_optimal_params(data):
    data[4] = normalize_score(data[4])
    data[5] = normalize_score(data[5])

    ind = np.argmax((data[4] + data[5]) / 2)

    return ind


def print_results(data, ana_params, optimized_parameters):
    ind = find_optimal_params(data)

    print('The following Parameters were the default:')
    pprint(vars(ana_params))

    print('\nThe following Combination of Paramters was the optimal one:')

    for i in range(data.shape[0] - 3):
        print(str(i) + ' Optimal ' + str(optimized_parameters[i]) + ' in Training: \t' + str(data[i, ind]))

    pass


def analysis(analysis, analysis_name):
    if str(analysis) == 'lr_ep_bs':
        data, ana_param, optimized_parameters = analyse_lr_nbep_bs(analysis_name)
        plot_lr_nbep_bs(analysis_name)
        print_results(data, ana_param, optimized_parameters)