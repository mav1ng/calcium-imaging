import data
import helpers as h

from os import listdir
from os.path import isfile, join

import json
import numpy as np
import torch
import network as n
import clustering as cl

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from torchvision import transforms, utils

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


def plot_analysis(analysis_name, th):
    ana_list = get_analysis(analysis_name=analysis_name)
    val_score_list = []

    for ana in ana_list:
        val_score_list.append(float(ana.val_score))
        if float(ana.val_score) >= th:
            print('Model Name: ', ana.model_name)
            print('Model Score: ', ana.val_score)
            dtype = torch.float
            device = torch.device('cuda:0')

            dic = data.read_from_json('config/' + str(ana.model_name) + '.json')

            model = n.UNetMS(input_channels=int(dic['input_channels']),
                             embedding_dim=int(dic['embedding_dim']),
                             use_background_pred=dic['background_pred'] == 'True',
                             nb_iterations=int(dic['nb_iterations']),
                             kernel_bandwidth=dic['kernel_bandwidth'],
                             step_size=float(dic['step_size']),
                             use_embedding_loss=dic['Embedding Loss'] == 'True',
                             margin=float(dic['margin']),
                             include_background=dic['Include Background'] == 'True',
                             scaling=float(dic['scaling']),
                             subsample_size=int(dic['subsample_size']))

            model.to(device)
            model.type(dtype)
            model.load_state_dict(torch.load('model/model_weights_' + str(ana.model_name) + '.pt'))

            test_dataset = data.TestCombinedDataset(corr_path='data/corr/starmy/maxpool/transformed_4/',
                                                    corr_sum_folder='data/corr_sum_img/',
                                                    sum_folder='data/sum_img/',
                                                    transform=None, device=device, dtype=dtype)

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)
            print('Test loader prepared.')

            model.eval()
            model.MS.test = True

            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    input = batch['image']

                    # compute output
                    if model.use_background_pred:
                        output, _, __ = model(input, None)
                    else:
                        output, _ = model(input, None)

                    (bs, ch, w, h) = output.size()

                    for i in range(ch):
                        if i % 10 == 0:
                            plt.imshow(output[0, i].detach().cpu().numpy())
                            plt.show()
                    break

            model.MS.val = False
            model.MS.test = False

    print('Highest Val Score: ', max(val_score_list))

    pass


def save_images(analysis_name, th=0., postproc=False):
    ana_list = get_analysis(analysis_name=analysis_name)
    val_score_list = []

    for ana in ana_list:
        val_score_list.append(float(ana.val_score))
        if float(ana.val_score) >= th:
            print('Model Name: ', ana.model_name)
            print('Model Score: ', ana.val_score)
            dtype = torch.float
            device = torch.device('cuda:0')

            dic = data.read_from_json('config/' + str(ana.model_name) + '.json')

            model = n.UNetMS(input_channels=int(dic['input_channels']),
                             embedding_dim=int(dic['embedding_dim']),
                             use_background_pred=dic['background_pred'] == 'True',
                             nb_iterations=int(dic['nb_iterations']),
                             kernel_bandwidth=dic['kernel_bandwidth'],
                             step_size=float(dic['step_size']),
                             use_embedding_loss=dic['Embedding Loss'] == 'True',
                             margin=float(dic['margin']),
                             include_background=dic['Include Background'] == 'True',
                             scaling=float(dic['scaling']),
                             subsample_size=int(dic['subsample_size']))

            model.to(device)
            model.type(dtype)
            model.load_state_dict(torch.load('model/model_weights_' + str(ana.model_name) + '.pt'))

            test_dataset = data.TestCombinedDataset(corr_path='data/corr/starmy/maxpool/transformed_4/',
                                                    corr_sum_folder='data/corr_sum_img/',
                                                    sum_folder='data/sum_img/',
                                                    transform=None, device=device, dtype=dtype)

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)
            print('Test loader prepared.')

            model.eval()
            model.MS.test = True

            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    input = batch['image']

                    # compute output
                    if model.use_background_pred:
                        output, _, background = model(input, None)
                    else:
                        output, _ = model(input, None)

                    (bs, ch, w, h) = output.size()

                    if torch.sum(torch.isnan(output)) > 0.:
                        print('Output contains NaN')
                        break

                    predict = cl.label_embeddings(output.view(ch, -1).t(), th=0.8)
                    predict = predict.reshape(bs, w, h)

                    if postproc:
                        if model.use_background_pred:
                            predict = cl.postprocess_label(predict, background=background[:, 0])
                        else:
                            predict = cl.postprocess_label(predict, background=None)

                    for b in range(bs):
                        plt.imshow(predict[b])
                        plt.title(str(ana.model_name))
                        if not postproc:
                            plt.savefig('images/' + str(ana.model_name) + '.png')
                        else:
                            plt.savefig('images_test/' + str(ana.model_name) + '.png')

                    break

            model.MS.val = False
            model.MS.test = False

    pass


def full_score_analysis(analysis_name, analysis_list, include_metric=True, iter=10):
    model_name_ = []
    f1_ = []
    f1_std_ = []
    rec_ = []
    rec_std_ = []
    prec_ = []
    prec_std_ = []
    emb_ = []
    emb_std_ = []
    cel_ = []
    cel_std_ = []

    for ana in analysis_list:
        print('Evaluating ' + str(ana.model_name))

        cl_th = 0.8
        pp_th = 0.
        obj_size = 18
        hole_size = 3

        if 'abram' in str(ana.model_name):
            cl_th = 0.1
            pp_th = 0.4
            obj_size = 10
            hole_size = 12
        elif 'azrael' in str(ana.model_name):
            cl_th = 0.1
            pp_th = 0.25
            obj_size = 35
            hole_size = 6
        elif 'eve' in str(ana.model_name):
            cl_th = 1.25
            pp_th = 0.25
            obj_size = 10
            hole_size = 16
        elif 'adam' in str(ana.model_name):
            cl_th = 0.75
            pp_th = 0.15
            obj_size = 20
            hole_size = 14
        elif 'noah' in str(ana.model_name):
            cl_th = 1.5
            pp_th = 0.175
            obj_size = 20
            hole_size = 20

        f1, f1_std, rec, rec_std, prec, prec_std, emb, emb_std, cel, cel_std = h.val_score(
            model_name=str(ana.model_name), iter=iter, use_metric=include_metric, return_full=True, cl_th=cl_th,
            pp_th=pp_th, obj_size=obj_size, holes_size=hole_size)

        model_name_.append(str(ana.model_name))
        f1_.append(f1)
        f1_std_.append(f1_std)
        rec_.append(rec)
        rec_std_.append(rec_std)
        prec_.append(prec)
        prec_std_.append(prec_std)
        emb_.append(emb)
        emb_std_.append(emb_std)
        cel_.append(cel)
        cel_std_.append(cel_std)

    ret = [model_name_, f1_, f1_std_, rec_, rec_std_, prec_, prec_std_, emb_, emb_std_, cel_, cel_std_]
    data.write_to_json(ret, path='data/model_scores/full_score_' + str(analysis_name) + '.json')

    return ret


def full_score(analysis_name, include_metric, iter=10):
    ana_list = get_analysis(analysis_name)
    score_list = full_score_analysis(analysis_name=analysis_name, analysis_list=ana_list, include_metric=include_metric,
                                     iter=iter)
    return score_list



def val_score_analysis(analysis_list, include_metric, iter=1):
    ret = {}
    for ana in analysis_list:
        print('Evaluating ' + str(ana.model_name))
        (val, emb, cel) = h.val_score(model_name=str(ana.model_name), iter=iter, th=0.8, use_metric=include_metric)
        ret[str(ana.model_name)] = (val, emb, cel)
    return ret


def val_score_metric_analysis(analysis_list, iter=1):
    ret = {}
    for ana in analysis_list:
        print('Evaluating ' + str(ana.model_name))
        (val, emb, cel) = h.val_score(model_name=str(ana.model_name), iter=iter, th=0.8, use_metric=True)
        ret[str(ana.model_name)] = (val, emb, cel)
    return ret


def score(analysis_name, include_metric, iter=1):
    ana_list = get_analysis(analysis_name)
    score_list = val_score_analysis(ana_list, include_metric=include_metric, iter=iter)

    for ana in ana_list:
        if include_metric:
            ana.val_score = score_list[ana.model_name][0]
        ana.emb_score = score_list[ana.model_name][1]
        ana.cel_score = score_list[ana.model_name][2]
        data.save_config_score(ana.model_name, ana.val_score, ana.emb_score, ana.cel_score, ana.input_channels,
                               ana.embedding_dim, ana.background_pred, ana.mean_shift_on, ana.nb_iterations,
                               ana.kernel_bandwidth, ana.step_size, ana.embedding_loss, ana.margin,
                               ana.include_background, ana.scaling, ana.subsample_size, ana.learning_rate,
                               ana.nb_epochs, ana.batch_size, ana.pre_train, ana.pre_train_name)
    pass


def score_metric(analysis_name):
    ana_list = get_analysis(analysis_name)
    score_list = val_score_metric_analysis(ana_list)

    for ana in ana_list:
        ana.val_score = score_list[ana.model_name][0]
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

    ret = np.where(ret == -1., np.nan, ret)

    return ret, ana_param, optimized_parameters


def analyse_ed_ma_sc(analysis_name):
    ana_list = get_analysis(analysis_name)

    ana_param = ana_list[0]

    ret = np.zeros((6, ana_list.__len__()))

    for i, ana in enumerate(ana_list):
        ret[0, i] = float(ana.embedding_dim)
        ret[1, i] = float(ana.margin)
        ret[2, i] = float(ana.scaling)
        ret[3, i] = float(ana.val_score)
        ret[4, i] = float(ana.emb_score)
        ret[5, i] = float(ana.cel_score)

    optimized_parameters = ['Embedding Dimension', 'Margin', 'Scaling']

    ret = np.where(ret == -1., np.nan, ret)

    return ret, ana_param, optimized_parameters


def analyse_ed_ma(analysis_name):
    ana_list = get_analysis(analysis_name)

    ana_param = ana_list[0]

    ret = np.zeros((5, ana_list.__len__()))

    for i, ana in enumerate(ana_list):
        ret[0, i] = float(ana.embedding_dim)
        ret[1, i] = float(ana.margin)
        ret[2, i] = float(ana.val_score)
        ret[3, i] = float(ana.emb_score)
        ret[4, i] = float(ana.cel_score)

    optimized_parameters = ['Embedding Dimension', 'Margin']

    ret = np.where(ret == -1., np.nan, ret)

    return ret, ana_param, optimized_parameters


def analyse_kb_iter(analysis_name):
    ana_list = get_analysis(analysis_name)

    ana_param = ana_list[0]

    ret = np.zeros((5, ana_list.__len__()))

    for i, ana in enumerate(ana_list):
        ret[0, i] = float(ana.kernel_bandwidth)
        ret[1, i] = float(ana.nb_iterations)
        ret[2, i] = float(ana.val_score)
        ret[3, i] = float(ana.emb_score)
        ret[4, i] = float(ana.cel_score)

    optimized_parameters = ['Kernel Bandwidth', 'Number of Iterations']

    ret = np.where(ret == -1., np.nan, ret)

    return ret, ana_param, optimized_parameters


def analyse_ss(analysis_name):
    ana_list = get_analysis(analysis_name)

    ana_param = ana_list[0]

    ret = np.zeros((4, ana_list.__len__()))

    for i, ana in enumerate(ana_list):
        ret[0, i] = float(ana.subsample_size)
        ret[1, i] = float(ana.val_score)
        ret[2, i] = float(ana.emb_score)
        ret[3, i] = float(ana.cel_score)

    optimized_parameters = ['Subsample Size']

    ret = np.where(ret == -1., np.nan, ret)

    return ret, ana_param, optimized_parameters


def analyse_lr(analysis_name):
    ana_list = get_analysis(analysis_name)

    ana_param = ana_list[0]

    ret = np.zeros((4, ana_list.__len__()))

    for i, ana in enumerate(ana_list):
        ret[0, i] = float(ana.learning_rate)
        ret[1, i] = float(ana.val_score)
        ret[2, i] = float(ana.emb_score)
        ret[3, i] = float(ana.cel_score)

    optimized_parameters = ['Learning Rate']

    ret = np.where(ret == -1., np.nan, ret)

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
    ret = np.where(d_n_ == 0., 0., d_n / d_n_)

    return ret


def plot_lr_nbep_bs(analysis_name, use_metric):
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

    if use_metric:
        c = data[3]
    else:
        c = ((data[4] + data[5]) / 2)

    img = ax.scatter(x, y, z, c=c, cmap=plt.cool())
    fig.colorbar(img)
    plt.ylim(0, 250)
    plt.show()

    plt.scatter(x, c)
    plt.show()
    plt.scatter(y, c)
    plt.xlim(0, 250)
    plt.show()
    plt.scatter(z, c)
    plt.show()


def plot_ed_ma_sc(analysis_name, use_metric):
    """
    Method that visualizes the Perfomance of a model depending on Embdding Dim, Margin and Scaling
    in a 4D graph by normalizing and weighting embedding and cel score perhaps not smart
    :param analysis_name:
    :return:
    """
    data, ana_param, _ = analyse_ed_ma_sc(analysis_name)

    data[4] = normalize_score(data[4])
    data[5] = normalize_score(data[5])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data[0]
    y = data[1]
    z = data[2]

    if use_metric:
        c = data[3]
    else:
        c = ((data[4] + data[5]) / 2)

    img = ax.scatter(x, y, z, c=c, cmap=plt.cool())
    fig.colorbar(img)
    plt.show()

    plt.scatter(x, c)
    plt.show()
    plt.scatter(y, c)
    plt.show()
    plt.scatter(z, c)
    plt.show()


def plot_ed_ma(analysis_name, use_metric):
    """
    Method that visualizes the Perfomance of a model depending on Embedding Dim an Margin
    in a 4D graph by normalizing and weighting embedding and cel score perhaps not smart
    :param analysis_name:
    :return:
    """
    data, ana_param, _ = analyse_ed_ma(analysis_name)

    data[3] = normalize_score(data[3])
    data[4] = normalize_score(data[4])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data[0]
    y = data[1]

    if use_metric:
        z = data[2]
    else:
        z = ((data[3] + data[4]) / 2)

    img = ax.scatter(x, y, z)
    plt.show()

    plt.scatter(x, z)
    plt.show()
    plt.scatter(y, z)
    plt.show()


def plot_kb_iter(analysis_name, use_metric):
    """
    Method that visualizes the Perfomance of a model depending on kernel bandwidth and number
    of iterations
    in a 4D graph by normalizing and weighting embedding and cel score perhaps not smart
    :param analysis_name:
    :return:
    """
    data, ana_param, _ = analyse_kb_iter(analysis_name)

    data[3] = normalize_score(data[3])
    data[4] = normalize_score(data[4])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data[0]
    y = data[1]

    if use_metric:
        z = data[2]
    else:
        z = ((data[3] + data[4]) / 2)

    img = ax.scatter(x, y, z)
    plt.show()

    plt.scatter(x, z)
    plt.show()
    plt.scatter(y, z)
    plt.show()


def plot_ss(analysis_name, use_metric):
    """
    Method that visualizes the Perfomance of a model depending on Subsample Size
    in a 4D graph by normalizing and weighting embedding and cel score perhaps not smart
    :param analysis_name:
    :return:
    """
    data, ana_param, _ = analyse_ss(analysis_name)

    data[2] = normalize_score(data[2])
    data[3] = normalize_score(data[3])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data[0]

    if use_metric:
        z = data[1]
    else:
        z = ((data[2] + data[3]) / 2)

    img = ax.scatter(x, z)
    plt.show()


def plot_lr(analysis_name, use_metric):
    """
    Method that visualizes the Perfomance of a model depending on Learning Rate
    in a 4D graph by normalizing and weighting embedding and cel score perhaps not smart
    :param analysis_name:
    :return:
    """
    data, ana_param, _ = analyse_lr(analysis_name)

    data[2] = normalize_score(data[2])
    data[3] = normalize_score(data[3])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data[0]

    if use_metric:
        z = data[1]
    else:
        z = ((data[2] + data[3]) / 2)

    img = ax.scatter(x, z)
    plt.show()


def find_optimal_params(data, score_ind, use_metric):
    (i, j) = score_ind

    if use_metric:
        ind = np.nanargmax((data[i - 1]))
    else:
        data[i] = normalize_score(data[i])
        data[j] = normalize_score(data[j])
        ind = np.nanargmin((data[i] + data[j]) / 2)

    return ind


def print_results(data, ana_params, optimized_parameters, score_ind, use_metric):
    ind = find_optimal_params(data, score_ind, use_metric)

    print('The following Parameters were the default:')
    pprint(vars(ana_params))

    print('\nThe following Combination of Parameters was the optimal one:')

    for i in range(data.shape[0] - 3):
        print(str(i) + ' Optimal ' + str(optimized_parameters[i]) + ' in Training: \t' + str(data[i, ind]))

    pass


def full_analyse(analysis_name, analysis):
    ana_list = get_analysis(analysis_name)
    scores = data.read_from_json('full_score_' + str(analysis_name) + '.json')
    name_list = scores[0]
    scores = np.array([scores[1:]])

    ret = np.zeros((11, ana_list.__len__()))

    for i, ana in enumerate(ana_list):
        current_model_name = name_list == str(ana.model_name)
        index = np.argwhere(np.array(current_model_name) == 1)
        cur_score = scores[index]

        if analysis =='ss':
            ret[0, i] = float(ana.subsample_size)
        elif analysis == 'scaling':
            ret[0, i] = float(ana.scaling)

        for j in range(11):
            ret[j + 1, i] = cur_score[j]

    optimized_parameters = ['Subsample Size']

    ret = np.where(ret == -1., np.nan, ret)

    return ret, optimized_parameters


def full_plot_ss(data):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data[0]
    y = data[9]
    y_err = data[10]

    plt.scatter(x, y, cmap='tab20b')
    ax.errorbars(x, y, y_err=y_err)
    plt.show()


def full_plot_scaling(data):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data[0]
    y = data[9]
    y_err = data[10]

    plt.scatter(x, y, cmap='tab20b')
    ax.errorbars(x, y, y_err=y_err)
    plt.show()


def full_analysis(analysis, analysis_name):
    if str(analysis) == 'ss':
        data, optimized_parameters = full_analyse(analysis_name, analysis='ss')
        full_plot_ss(data)
    elif str(analysis) == 'scaling':
        data, optimized_parameters = full_analyse(analysis_name, analysis='scaling')
        full_plot_scaling(data)

    else:
        print('Analysis Name is not known!')

def analysis(analysis, analysis_name, use_metric):
    if str(analysis) == 'lr_ep_bs':
        data, ana_param, optimized_parameters = analyse_lr_nbep_bs(analysis_name)
        plot_lr_nbep_bs(analysis_name, use_metric)
        print_results(data, ana_param, optimized_parameters, (4, 5), use_metric)
    elif str(analysis) == 'ed_ma_sc':
        data, ana_param, optimized_parameters = analyse_ed_ma_sc(analysis_name)
        plot_ed_ma_sc(analysis_name, use_metric)
        print_results(data, ana_param, optimized_parameters, (4, 5), use_metric)
    elif str(analysis) == 'ed_ma':
        data, ana_param, optimized_parameters = analyse_ed_ma(analysis_name)
        plot_ed_ma(analysis_name, use_metric)
        print_results(data, ana_param, optimized_parameters, (3, 4), use_metric)
    elif str(analysis) == 'ss':
        data, ana_param, optimized_parameters = analyse_ss(analysis_name)
        plot_ss(analysis_name, use_metric)
        print_results(data, ana_param, optimized_parameters, (2, 3), use_metric)
    elif str(analysis) == 'lr':
        data, ana_param, optimized_parameters = analyse_lr(analysis_name)
        plot_lr(analysis_name, use_metric)
        print_results(data, ana_param, optimized_parameters, (2, 3), use_metric)
    elif str(analysis) == 'kb_iter':
        data, ana_param, optimized_parameters = analyse_kb_iter(analysis_name)
        plot_kb_iter(analysis_name, use_metric)
        print_results(data, ana_param, optimized_parameters, (3, 4), use_metric)

    else:
        print('Analysis Name is not known!')


def input_test(nb_neuro, input_dim, corr_path, corr_sum_folder, sum_folder, show_label):
    # transform_train = transforms.Compose([data.RandomCrop(128)])
    # val_dataset = data.CombinedDataset(corr_path='data/corr/starmy/maxpool/transformed_4/',
    #                                    corr_sum_folder='data/corr_sum_img/',
    #                                    sum_folder='data/sum_img/',
    #                                    mask_folder='data/sum_masks/',
    #                                    transform=None, device=device, dtype=dtype, test=True)
    train_dataset = data.CombinedDataset(corr_path=corr_path,
                                         corr_sum_folder=corr_sum_folder,
                                         sum_folder=sum_folder,
                                         mask_folder='data/sum_masks/',
                                         transform=None, device=torch.device('cuda:0'), dtype=torch.float, test=False)
    val_dataset = data.CombinedDataset(corr_path=corr_path,
                                       corr_sum_folder=corr_sum_folder,
                                       sum_folder=sum_folder,
                                       mask_folder='data/sum_masks/',
                                       transform=None, device=torch.device('cuda:0'), dtype=torch.float, test=True)

    if not show_label:
        cur_img = torch.cat([train_dataset[nb_neuro]['image'][input_dim], val_dataset[nb_neuro]['image'][input_dim]], dim=1)
    else:
        cur_img = torch.cat([train_dataset[nb_neuro]['image'][input_dim], val_dataset[nb_neuro]['image'][input_dim]],
                            dim=1)
        lab_img = torch.cat([train_dataset[nb_neuro]['label'], val_dataset[nb_neuro]['label']], dim=1)

        cur_img = torch.where(lab_img == 1., torch.max(cur_img), cur_img)

    return cur_img


def input_labels(nb_neuro='data/training_data/', diff_labs=True, show_images=True, save_images=False):
    dtype = torch.float
    device = torch.device('cuda:0')

    train_dataset = data.CombinedDataset(corr_path='data/corr/starmy/maxpool/transformed_4/',
                                         corr_sum_folder='data/corr_sum_img/',
                                         sum_folder='data/sum_img/',
                                         mask_folder='data/sum_masks/',
                                         transform=None, device=device, dtype=dtype, train_val_ratio=1.0)

    for i, ds in enumerate(train_dataset):
        cur_img = ds['label'].detach().cpu().numpy()

        if diff_labs:
            cur_img = np.squeeze(h.get_diff_labels(np.expand_dims(cur_img, axis=0), background=0), axis=0)

        if show_images:
            plt.imshow(cur_img, cmap='tab20b')
            plt.show()

        if save_images:
            plt.imshow(cur_img, cmap='tab20b')
            plt.savefig('x_images/labels/td_' + str(i) + '.pdf')


def show_input(image, image_name, save_image):
    plt.imshow(image.detach().cpu().numpy(), cmap='gray')
    if save_image:
        plt.savefig('x_images/' + str(image_name) + '.pdf')
    # plt.show()
