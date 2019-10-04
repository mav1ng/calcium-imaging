import json

import numpy as np
import clustering as cl
import skimage.measure as skm


from torch import optim
import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter
import umap
import torch.nn as nn
import config as c
import data
import corr
import network as n
import visualization as v
import training as t
import clustering as cl
import helpers as h
import argparse
import time
import random
import numpy as np
import neurofinder as nf
from PIL import Image

from skimage import morphology


class Setup:
    def __init__(self, model_name, save_config, input_channels=c.UNet['input_channels'],
                embedding_dim=c.UNet['embedding_dim'],
                background_pred=c.UNet['background_pred'],
                mean_shift_on=c.mean_shift['nb_iterations'] > 0,
                nb_iterations=c.mean_shift['nb_iterations'],
                kernel_bandwidth=c.mean_shift['kernel_bandwidth'],
                step_size=c.mean_shift['step_size'],
                embedding_loss=c.embedding_loss['on'],
                margin=c.embedding_loss['margin'],
                include_background=c.embedding_loss['include_background'],
                scaling=c.embedding_loss['scaling'],
                subsample_size=c.embedding_loss['subsample_size'],
                learning_rate=c.training['lr'],
                nb_epochs=c.training['nb_epochs'],
                batch_size=c.training['batch_size'],
                pre_train=c.tb['pre_train'],
                pre_train_name=c.tb['pre_train_name'],
                th_nn=c.val['th_nn']):

        self.th_nn = th_nn
        self.writer = SummaryWriter(log_dir='training_log/' + str(model_name) + '/')
        self.device = torch.device('cuda:0')
        self.nb_cpu_threads = c.cuda['nb_cpu_threads']
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

        self.breaking = False

        if save_config:
            data.save_config(model_name=model_name, input_channels=input_channels, embedding_dim=embedding_dim,
                             background_pred=background_pred,
                             mean_shift_on=mean_shift_on, nb_iterations=nb_iterations, kernel_bandwidth=kernel_bandwidth,
                             step_size=step_size,
                             embedding_loss=embedding_loss, margin=margin, include_background=include_background,
                             scaling=scaling,
                             subsample_size=subsample_size, learning_rate=learning_rate, nb_epochs=nb_epochs,
                             pre_train=pre_train,
                             pre_train_name=pre_train_name, batch_size=batch_size)


    def train(self, train_loader, model, criterion, criterionCEL, optimizer, epoch):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        emb_losses = AverageMeter()
        cel_losses = AverageMeter()

        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for i, batch in enumerate(train_loader):
            input = batch['image']
            label = batch['label']

            if torch.sum(torch.isnan(input)) > 0.:
                print('Nan in Input')
                for i in range(input.size(0)):
                    print(input[i])

            # measure data loading time
            data_time.update(time.time() - end)

            input_var = list()
            for j in range(len(input)):
                input_var.append(torch.autograd.Variable(input[j]))

            target_var = list()
            for j in range(len(label)):
                target_var.append(torch.autograd.Variable(label[j]))

            input.requires_grad = True
            label.requires_grad = True

            if c.debug['print_input']:
                v.plot_emb_pca(input[0].detach(), label.detach())
                v.plot_input(input[0].detach(), label.detach())

            # zero the parameter gradients
            optimizer.zero_grad()

            if self.background_pred:
                output, ret_loss, y = model(input, label)

                '''Cross Entropy Loss on Background Prediction'''
                cel_loss = torch.tensor(0., device=self.device)
                for b in range(y.size(0)):
                    lab = torch.where(label[b].flatten().long() > 0,
                                      torch.tensor(1, dtype=torch.long, device=self.device),
                                      torch.tensor(0, dtype=torch.long, device=self.device))
                    cel_loss = cel_loss.clone() + criterionCEL(y[b].view(2, -1).t(), lab)

                    # if cel_loss != cel_loss:
                    #     print(y)
                    #     print(torch.sum(torch.isnan(y)))

                cel_loss = cel_loss / y.size(0)

                self.writer.add_scalar('Cross Entropy Loss', cel_loss.item())

                cel_loss.backward()
                cel_losses.update(cel_loss.item())

                if c.debug['print_img']:
                    v.plot_pred_back(y[0].detach(), label[0].detach())
            else:
                output, ret_loss = model(input, label)
                cel_loss = torch.tensor(0., device=self.device)

            # measure performance and record loss
            try:
                emb_losses.update(ret_loss.item())
            except AttributeError:
                emb_losses.update(ret_loss)




            #
            #
            # print(output.size())
            # print(label.size())
            #
            # data_ = output[0].view(3, -1).t().detach().cpu().numpy()
            # label_ = get_diff_labels(label.detach().cpu().numpy(), background=0)[0].reshape(-1)
            #
            # # plt.scatter(data_[:, 0], data_[:, 1], data_[:, 2], c=label_, cmap='tab20b')
            # # plt.show()
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(data_[:, 0], data_[:, 1], data_[:, 2], c=label_, cmap='tab20b', s=5, alpha=1)
            #
            # plt.axis('off')
            #
            # u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:50j]
            # x = np.cos(u) * np.sin(v)
            # y = np.sin(u) * np.sin(v)
            # z = np.cos(v)
            # ax.plot_wireframe(x, y, z, cmap='tab20b', color='grey', alpha=0.3)
            #
            # plt.show()





            if c.debug['print_img']:
                # fig = v.draw_umap(data=output[0].detach().view(c.UNet['embedding_dim'], -1),
                #                   color=label[0].detach().flatten())
                # plt.show()

                # pred_labels = cl.label_embeddings(output[0].view(self.embedding_dim, -1).t().detach(),
                #                                   th=c.val['th_nn'])
                # pred_labels2 = cl.label_emb_sl(output[0].view(self.embedding_dim, -1).t().detach(),
                #                                th=c.val['th_sl'])
                #
                # print('There are ' + str(torch.unique(label).size(0)) + ' clusters.')
                #
                # v.plot_sk_img(pred_labels, label.detach())
                # v.plot_sk_img(pred_labels2, label.detach())

                v.plot_emb_pca(output[0].detach().cpu().numpy(), label.detach().cpu().numpy())

            self.writer.add_scalar('Embedding Loss', ret_loss)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: {0}\t'
                  'Emb Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CEL Loss {lossCEL.val:.4f} ({lossCEL.avg:.4f})\t'
                  'Param ({param}))\t'.format(
                epoch, loss=emb_losses, lossCEL=cel_losses, param=optimizer.param_groups[0]['lr']))

            if ret_loss != ret_loss or cel_loss != cel_loss:
                print('Exploding Gradients! Stopped Training.')
                self.breaking = True
                break


    def validate(self, val_loader, model, use_metric, criterionCEL, scoring=False, pp_th=0., cl_th=0.8, return_full=False, obj_size=18, holes_size=3):

        # nf_threshold = c.val['nf_threshold']
        # switch to evaluate mode
        model.eval()
        model.MS.val = True
        model_name = self.model_name

        '''PARAMETERS FOR EMBEDDING LOSS ARE FIXED IN NETWORK FORWARD'''

        with torch.no_grad():

            end = time.time()

            f1_metric = 0.
            total_CEL_loss = 0.
            total_EMB_loss = 0.
            (recall, precision) = (0., 0.)

            for i, batch in enumerate(val_loader):
                input = batch['image']
                label = batch['label']

                input_var = list()
                for j in range(len(input)):
                    input_var.append(torch.autograd.Variable(input[j]))

                target_var = list()
                for j in range(len(label)):
                    target_var.append(torch.autograd.Variable(label[j]))

                cel_loss = torch.tensor(0., device=self.device)

                # compute output
                if self.background_pred:

                    # plt.imshow(input[0, 0].detach().cpu().numpy())
                    # plt.show()
                    # plt.imshow(label[0].detach().cpu().numpy())
                    # plt.show()

                    output, val_loss, y = model(input, label)

                    '''LOSS CALCULATION'''
                    '''Cross Entropy Loss on Background Prediction'''
                    for b in range(y.size(0)):
                        lab = torch.where(label[b].flatten().long() > 0,
                                          torch.tensor(1, dtype=torch.long, device=self.device),
                                          torch.tensor(0, dtype=torch.long, device=self.device))
                        cel_loss = cel_loss.clone() + criterionCEL(y[b].view(2, -1).t(), lab)
                    cel_loss = cel_loss / y.size(0)
                else:
                    output, val_loss = model(input, label)

                (bs, ch, w, h) = output.size()

                # for i in range(ch):
                #     plt.imshow(output[0, i].detach().cpu().numpy())
                #     plt.show()

                total_CEL_loss += cel_loss
                total_EMB_loss += val_loss

                if use_metric and torch.sum(torch.isnan(output)).item() == 0:
                    # predict = cl.label_emb_sl(output.view(ch, -1).t(), th=.5)

                    predict = cl.label_embeddings(output.view(ch, -1).t(), th=cl_th)
                    predict = predict.reshape(bs, w, h)

                    if scoring:
                        if model.use_background_pred:
                            predict = cl.postprocess_label(predict, background=y[:, 1], th=pp_th, hole_size=holes_size,
                                                           obj_size=obj_size)
                        else:
                            predict = cl.postprocess_label(predict, background=None, th=pp_th, hole_size=holes_size,
                                                           obj_size=obj_size)

                    f1_metric_ = 0.
                    (recall_, precision_) = (0., 0.)

                    label = get_diff_labels(label.cpu().numpy())

                    for b in range(bs):
                        self.writer.add_image('Prediction Epoch ' + str(self.epoch), predict[b].reshape(1, w, h),
                                              global_step=1)

                    if c.val['show_img']:
                        for b in range(bs):
                            f, axarr = plt.subplots(2)
                            axarr[0].imshow(predict[b])
                            axarr[1].imshow(label[b])

                            plt.title('Predicted Background (upper) vs Ground Truth (lower)')
                            plt.show()

                    for k in range(bs):
                        (mask_true, mask_predict) = (data.toCoords(label[k]), data.toCoords(predict[k]))
                        data.write_to_json(mask_true, 'data/val_mask/mask_true_' + str(model_name) + '.json')
                        data.write_to_json(mask_predict, 'data/val_mask/mask_predict_' + str(model_name) + '.json')
                        true_mask = nf.load('data/val_mask/mask_true_' + str(model_name) + '.json')
                        predict_mask = nf.load('data/val_mask/mask_predict_' + str(model_name) + '.json')
                        recall_c, precision_c = nf.centers(true_mask, predict_mask)
                        recall_ += recall_c
                        precision_ += precision_c
                        f1_metric_ = f1_metric_ + 2 * (recall_ * precision_) / (recall_ + precision_)

                    f1_metric_ = f1_metric_ / bs
                    recall_ = recall_ / bs
                    precision_ = precision_ / bs

                    f1_metric = f1_metric + f1_metric_
                    recall = recall + recall_
                    precision = precision + precision_

            count = val_loader.__len__()

            f1_metric = f1_metric / count
            recall = recall / count
            precision = precision / count

            ret_cel = cel_loss.item() / bs
            ret_emb = val_loss/ bs

            self.writer.add_scalar('Validation CEL', ret_cel)
            self.writer.add_scalar('Validation EMB', ret_emb)
            print('Average Validation Loss: EMB: ' + str(ret_emb) + '\tCEL: ' + str(ret_cel))

            print('* F1 Metric {f1:.4f}\t'
                  'Recall {recall}\t'
                  'Precision {precision}\t'.format(f1=f1_metric, recall=recall, precision=precision))

            model.MS.val = False

            self.writer.add_scalar('F1', f1_metric)
            self.writer.add_scalar('Recall', recall)
            self.writer.add_scalar('Precision', precision)

        if return_full:
            return f1_metric, recall, precision, ret_emb, ret_cel
        else:
            return f1_metric, ret_emb, ret_cel


    def main(self):

        dtype = torch.float
        device = torch.device('cuda:0')
        device_ids = c.cuda['use_devices']

        batch_size = self.batch_size
        lr = self.learning_rate
        nb_epochs = self.nb_epochs
        pre_train = self.pre_train

        resume = c.training['resume']
        start_epoch = 0
        img_size = c.training['img_size']
        num_workers = c.data['num_workers']
        nb_samples = c.training['nb_samples']
        val_freq = c.val['val_freq']

        torch.set_num_threads(self.nb_cpu_threads)

        model = n.UNetMS(input_channels=self.input_channels, embedding_dim=self.embedding_dim,
                         kernel_bandwidth=self.kernel_bandwidth,
                         margin=self.margin, step_size=self.step_size, nb_iterations=self.nb_iterations,
                         use_embedding_loss=self.embedding_loss, scaling=self.scaling,
                         use_background_pred=self.background_pred, subsample_size=self.subsample_size,
                         include_background=self.include_background)

        model.to(device)
        model.type(dtype)

        # loading old weights
        try:
            if self.pre_train:
                model.load_state_dict(torch.load('model/model_weights_' + str(self.pre_train_name) + '.pt'))
            else:
                model.load_state_dict(torch.load('model/model_weights_' + str(self.model_name) + '.pt'))
            print('Loaded Model!')
        except IOError:
            print('No Model to load from!')
            print('Initializing Weights and Bias')
            model.apply(t.weight_init)
            print('Finished Initializing')

        # define loss function (criterion) and optimizer
        # perhaps add additional loss functions if this one does not work

        optimizer = optim.Adam(model.parameters(), lr=lr)

        criterion = n.EmbeddingLoss(margin=self.margin).cuda()
        criterionCEL = nn.CrossEntropyLoss().cuda()

        # # creating different parameter groups
        base_params = list(map(id, model.parameters()))

        if resume:
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                best_val = checkpoint['best_val']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resume))
                best_f1 = 0.
        else:
            best_f1 = 0.

        # models are save only when their loss obtains the best value in the validation
        f1track = 0

        # data preparation, loaders
        # cudnn.benchmark = True

        # preparing the training loader
        dim_of_corrs = [2, 3, 4, 5]
        # dim_of_corrs = [3, 4, 5, 6]

        transform_train = transforms.Compose([data.RandomCrop(img_size),
                                              data.RandomRot(dim_of_corrs),
                                              data.RandomFlip(True, dim_of_corrs, 0.5),
                                              data.RandomFlip(False, dim_of_corrs, 0.5)
                                              ])

        train_dataset = data.CombinedDataset(corr_path='data/corr/starmy/maxpool/transformed_4/',
                                             corr_sum_folder='data/corr_sum_img/',
                                             sum_folder='data/sum_img/',
                                             mask_folder='data/sum_masks/',
                                             transform=transform_train, device=device, dtype=dtype)

        random_sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True,
                                                        num_samples=(nb_samples * batch_size))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=random_sampler,
                                                   num_workers=num_workers)
        print('Training loader prepared.')

        # preparing validation loader
        transform_val = transforms.Compose([data.RandomCrop(img_size, val=True)])
        val_dataset = data.CombinedDataset(corr_path='data/corr/starmy/maxpool/transformed_4/',
                                           corr_sum_folder='data/corr_sum_img/',
                                           sum_folder='data/sum_img/',
                                           mask_folder='data/sum_masks/',
                                           transform=None, device=device, dtype=dtype, test=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=num_workers)
        print('Validation loader prepared.')

        # run epochs
        for epoch in range(start_epoch, nb_epochs):

            self.epoch = epoch

            # train for one epoch
            self.train(train_loader, model, criterion, criterionCEL, optimizer, epoch)

            # evaluate on validation set
            if (epoch + 1) % val_freq == 0 and epoch != 0:
                'rewrite THIS PART BECAUSE USING NEUROFINDER METRIC'
                f1_metric, _, __ = self.validate(val_loader, model, criterionCEL=criterionCEL, use_metric=False)

                # check patience
                if f1_metric <= best_f1:
                    f1track += 1
                else:
                    f1track = 0

                # save the best model
                is_best = f1_metric > best_f1
                best_f1 = max(f1_metric, best_f1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val': best_f1,
                    'optimizer': optimizer.state_dict(),
                    'valtrack': f1track,
                    'curr_val': f1_metric,
                }, is_best, model_name=self.model_name)

                print('** Validation: %f (best) - %d (f1track)' % (best_f1, f1track))

            print('Saved Model After Epoch')
            torch.save(model.state_dict(), 'model/model_weights_' + str(self.model_name) + '.pt')

            if self.breaking:
                break


def save_checkpoint(state, is_best, model_name, filename='checkpoint.pth.tar'):
    snapshots = c.data['snapshots']
    filename = snapshots + model_name +'_e%03d_v-%.3f.pth.tar' % (state['epoch'], state['best_val'])
    if is_best:
        torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_diff_labels(mask, background=0):
    """
    returns mask with different labels
    :param mask: input np array B x w x h
    :return: mask with a different label for every neuron
    """
    (bs, w, h) = mask.shape

    for b in range(bs):
        mask[b] = skm.label(mask[b], background=background)

    return mask

def get_input_diag(part_nb, dataset):
    """method that returns a sliced image along the diagonal from the dataset"""

    comb_dataset = dataset
    tn = part_nb
    input = comb_dataset[0]['image'][:, tn * 64:(tn + 1) * 64, tn * 64:(tn + 1) * 64].view(1, 10, 64, 64).cuda()
    label = comb_dataset[0]['label'][tn * 64:(tn + 1) * 64, tn * 64:(tn + 1) * 64].view(1, 64, 64).cuda()

    return input, label


def denoise(img, weight=0.1, eps=1e-3, num_iter_max=10000):
    """Perform total-variation denoising on a grayscale image.

    Parameters
    ----------
    img : array
        2-D input data to be de-noised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more
        de-noising (at the expense of fidelity to `img`).
    eps : float, optional
        Relative difference of the value of the cost
        function that determines the stop criterion.
        The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    num_iter_max : int, optional
        Maximal number of iterations used for the
        optimization.

    Returns
    -------
    out : array
        De-noised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """

    u = np.zeros_like(img)
    px = np.zeros_like(img)
    py = np.zeros_like(img)

    nm = np.prod(img.shape[:2])
    tau = 0.125

    i = 0
    while i < num_iter_max:
        u_old = u

        # x and y components of u's gradient
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u

        # update the dual variable
        px_new = px + (tau / weight) * ux
        py_new = py + (tau / weight) * uy

        norm_new = np.maximum(1, np.sqrt(px_new ** 2 + py_new ** 2))
        px = px_new / norm_new
        py = py_new / norm_new

        # calculate divergence
        rx = np.roll(px, 1, axis=1)
        ry = np.roll(py, 1, axis=0)
        div_p = (px - rx) + (py - ry)

        # update image
        u = img + weight * div_p

        # calculate error
        error = np.linalg.norm(u - u_old) / np.sqrt(nm)

        if i == 0:
            err_init = error
            err_prev = error
        else:
            # break if error small enough
            if np.abs(err_prev - error) < eps * err_init:
                break
            else:
                e_prev = error

        # don't forget to update iterator
        i += 1
    return u


def pad_nf(array, img_size=512, labels=False):
    """
    Padding the input array
    :param array: T x W x H , T time series, width , height
    :param img_size: wanted image size W x H
    :return: padded array
    """

    w = array.shape[-2]
    h = array.shape[-1]

    assert w <= img_size and h <= img_size, 'Image cannot be padded to be smaller'

    if w == img_size and h == img_size:
        return array

    w_ = w % 2
    h_ = h % 2

    w_pad = int((img_size - w) / 2)
    h_pad = int((img_size - h) / 2)

    if not labels:
        return np.pad(array, ((0, 0), (w_pad + w_, w_pad), (h_pad + h_, h_pad)), 'constant', constant_values=0)
    else:
        return np.pad(array, ((w_pad + w_, w_pad), (h_pad + h_, h_pad)), 'constant', constant_values=0)


def emb_subsample(embedding_tensor, label_tensor, backpred, include_background, prefer_cell, sub_size=100):
    """
    Method that returns a subsampled amount of pixel to calculate the embedding loss
    :param embedding_tensor: Bs x Ch x w x h
    :param label_tensor: Bs x w x h, zeroes and ones
    :param backpred: Bs x w x h, specifies probabilities of getting chosen
    :param include_background: Boolean whether to allow background pixels to be in the subsample
    :param prefer_cell: value in [0, 1] on how strongly to prefer foreground pixels, 0 no preference, 1 full preference
    :param sub_size: Parameter that specifies the number of pixels in subsampled image
    :return: subsampled embedding and labels
    """

    assert np.sqrt(sub_size) % 1 == 0, 'Sub Size needs to be a valid Image size, e.g 10 x 10 = 100'

    device = torch.device('cuda:0')

    (bs, ch, w, h) = embedding_tensor.size()
    (new_w, new_h) = (int(np.sqrt(sub_size)), int(np.sqrt(sub_size)))

    back_prob = torch.ones(bs, w, h, device=device) * 2
    if backpred is not None:
        softmax_layer = nn.Softmax2d()
        backpred = softmax_layer(backpred)

        # plt.imshow(backpred[0, 0].detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()
        # plt.imshow(backpred[0, 0].detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()

        back_prob = 1 - backpred[:, 0] * prefer_cell

    emb = torch.zeros(bs, ch, new_w, new_h, device=device)
    lab = torch.zeros(bs, new_w, new_h, device=device)

    for b in range(bs):

        sub_pool_size = 2
        while True:
            ind = torch.unique(torch.randint(0, w * h, (sub_size * sub_pool_size,), device=device))
            drop = torch.rand(ind.size(), device=device)
            prob = back_prob[b].view(-1)[ind]
            ind = torch.where(prob > drop, ind, torch.tensor(-1, device=device))

            if not include_background:
                ind = torch.where(label_tensor[b].view(-1)[ind] == 0., torch.tensor(-1, device=device), ind)

            ind = torch.where(ind == -1, torch.randint_like(ind, 0, w * h), ind)
            ind = torch.unique(ind)
            ind = ind[torch.randint(0, ind.size(0), (ind.size(0),))]

            if ind.size(0) > sub_size:
                ind = ind.clone()[:sub_size]
                break

            sub_pool_size += 1

            # Breaking Condition/Threshold
            if sub_size * sub_pool_size >= w * h:
                print('Not enough Cell pixels in input image, filling up with background pixels! '
                      '\nTaking random sub pixels!')

                while True:
                    ind = torch.unique(torch.randint(0, w * h, (sub_size * 2,), device=device))
                    if ind.size(0) > sub_size:
                        ind = ind.clone()[:sub_size]
                        break
                break

        emb[b] = embedding_tensor[b].view(ch, -1)[:, ind].view(ch, new_w, new_h)
        lab[b] = label_tensor[b].view(-1)[ind].view(new_w, new_h)

    return emb, lab, ind


def test(model_name, cl_th, pp_th, obj_size, hole_size, show_image=True, save_image=False):

    dtype = torch.float
    device = torch.device('cuda:0')

    dic = data.read_from_json('config/' + str(model_name) + '.json')

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
    model.load_state_dict(torch.load('model/model_weights_' + str(model_name) + '.pt'))

    print('Testing ' + str(model_name))

    results = []
    namelist = ['00.00.test', '00.01.test', '01.00.test', '01.01.test', '02.00.test', '02.01.test', '03.00.test',
                '04.00.test', '04.01.test']

    test_dataset = data.TestCombinedDataset(corr_path='data/test_corr/starmy/maxpool/transformed_4/',
                                            corr_sum_folder='data/test_corr_sum_img/',
                                            sum_folder='data/test_sum_img/',
                                            transform=None, device=device, dtype=dtype)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)
    print('Test loader prepared.')

    model.eval()
    model.MS.test = True

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print(i)

            input = batch['image']
            input_var = list()
            for j in range(len(input)):
                input_var.append(torch.autograd.Variable(input[j]))


            # compute output
            if model.use_background_pred:
                output, _, background = model(input, None)
            else:
                output, _ = model(input, None)

            (bs, ch, w, h) = output.size()

            # for k in range(ch):
            #     plt.imshow(output[0, k].detach().cpu().numpy())
            #     plt.show()

            # plt.imshow(__[0, 0].detach().cpu().numpy())
            # plt.show()

            # labels = np.ones((bs, w, h))
            # v.plot_emb_pca(output[0], labels[0])

            # pos_matrix = cl.get_pos_mat(bs, w, h)
            # m = torch.cat([output, pos_matrix], dim=1)
            #
            # predict = cl.label_embeddings(m.view(ch + 2, -1).t(), th=0.8)
            # predict = predict.reshape(bs, w, h)

            predict = cl.label_embeddings(output.view(ch, -1).t(), th=cl_th)
            predict = predict.reshape(bs, w, h)

            # if i == 0:
            #     pp_th = 0.2
            # elif i == 1:
            #     pp_th = 0.19
            # elif i == 2:
            #     pp_th = 0.19
            # elif i == 3:
            #     pp_th = 0.2
            # elif i == 4:
            #     pp_th = 0.21
            # elif i == 5:
            #     pp_th = 0.215
            # elif i == 6:
            #     pp_th = 0.14
            # elif i == 7:
            #     pp_th = 0.22
            # elif i == 8:
            #     pp_th = 0.21
            #
            # print(pp_th)

            if model.use_background_pred:
                predict = cl.postprocess_label(predict, background=background[:, 1], embeddings=output, th=pp_th,
                                               obj_size=obj_size, hole_size=hole_size)
            else:
                predict = cl.postprocess_label(predict, background=None, obj_size=obj_size, hole_size=hole_size)

            if show_image:
                plt.imshow(predict[0], cmap='tab20b')
                plt.title('Predicted Background (upper) vs Ground Truth (lower)')
                plt.show()

            if save_image:
                plt.imshow(predict[0], cmap='tab20b')
                plt.savefig('x_images/output/' + str(model_name) + '_testd_' + str(i) + '.pdf')

            mask_predict = data.toCoords(predict[0])
            results.append({'dataset': namelist[i], 'regions': mask_predict})

        if not os.path.exists('data/test_results'):
            os.makedirs('data/test_results')

        with open('data/test_results/' + str(model_name) + '.json', 'w') as f:
            f.write(json.dumps(results))

        model.MS.val = False
        model.MS.test = False

    pass


def create_output_image(model_name, cl_th, pp_th, obj_size, hole_size, show_image=False, save_images=False):

    dtype = torch.float
    device = torch.device('cuda:0')

    dic = data.read_from_json('config/' + str(model_name) + '.json')

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
    model.load_state_dict(torch.load('model/model_weights_' + str(model_name) + '.pt'))

    print('Testing ' + str(model_name))

    test_dataset = data.CombinedDataset(corr_path='data/corr/starmy/maxpool/transformed_4/',
                                       corr_sum_folder='data/corr_sum_img/',
                                       sum_folder='data/sum_img/',
                                        mask_folder='data/sum_masks/',
                                       transform=None, device=device, dtype=dtype, train_val_ratio=1.0)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)
    print('Test loader prepared.')

    model.eval()
    model.MS.test = True

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print(i)

            input = batch['image']
            input_var = list()
            for j in range(len(input)):
                input_var.append(torch.autograd.Variable(input[j]))


            # compute output
            if model.use_background_pred:
                output, _, background = model(input, None)
            else:
                output, _ = model(input, None)

            (bs, ch, w, h) = output.size()

            predict = cl.label_embeddings(output.view(ch, -1).t(), th=cl_th)
            predict = predict.reshape(bs, w, h)

            if model.use_background_pred:
                predict = cl.postprocess_label(predict, background=background[:, 1], embeddings=output, th=pp_th,
                                               obj_size=obj_size, hole_size=hole_size)
            else:
                predict = cl.postprocess_label(predict, background=None, obj_size=obj_size, hole_size=hole_size)

            if show_image:
                plt.imshow(predict[0], cmap='tab20b')
                plt.title('Predicted Background (upper) vs Ground Truth (lower)')
                plt.show()

            if save_images:
                plt.imshow(predict[0], cmap='tab20b')
                plt.savefig('x_images/output/' + str(model_name) + '_td_' + str(i) + '.pdf')

        model.MS.val = False
        model.MS.test = False

    pass



def find_th(model_name, iter=10):
    best_score = 0.
    (best_th_cl, best_th_pp) = (0., 0.)
    th_list = []
    bp_th_list = []
    score_list = []
    for th in [0.1, 0.5, 0.75, 1., 1.25, 1.5]:
        for back_pred in [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]:
            print('Current ClTh: ' + str(th) + '\t Current pp th: ' + str(back_pred))
            cur_met = val_score(model_name, use_metric=True, iter=iter, cl_th=th, pp_th=back_pred)[0]
            if best_score < cur_met:
                best_score = cur_met
                (best_th_cl, best_th_pp) = (th, back_pred)
            th_list.append(th)
            bp_th_list.append(back_pred)
            score_list.append(cur_met)
    print('For model ' + str(model_name) + ' the best thresholds were: \n cl_th: ' + str(best_th_cl) + '\n pp_th: ' + str(
            best_th_pp))

    ret = np.array([th_list, bp_th_list, score_list])
    data.save_numpy_to_h5py(ret, model_name + '_find_th', file_path='plot_data/')

    return best_th_cl, best_th_pp


def find_optimal_object_size(model_name, cl_th, pp_th, iter=10):
    best_score = 0.
    best_object_size = 0
    o_list = []
    score_list = []
    for obj in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 120, 150]:
        print('Current Object Size: ' + str(obj))
        cur_met = \
        val_score(model_name, use_metric=True, iter=iter, cl_th=cl_th, pp_th=pp_th, obj_size=obj, holes_size=3)[0]
        if best_score < cur_met:
            best_score = cur_met
            best_object_size = obj
        o_list.append(obj)
        score_list.append(cur_met)

    ret = np.array([o_list, score_list])
    data.save_numpy_to_h5py(ret, model_name + '_find_obj_size', file_path='plot_data/')

    best_score = 0.
    best_hole_size = 0
    h_list = []
    score_list = []
    for hole in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]:
        print('Current Hole Size: ' + str(hole))
        cur_met = val_score(model_name, use_metric=True, iter=iter, cl_th=cl_th, pp_th=pp_th, obj_size=best_object_size,
                            holes_size=hole)[0]
        if best_score < cur_met:
            best_score = cur_met
            best_hole_size = hole
        h_list.append(hole)
        score_list.append(cur_met)

    ret2 = np.array([h_list, score_list])
    data.save_numpy_to_h5py(ret2, model_name + '_find_hole_size', file_path='plot_data/')

    print('For model ' + str(model_name) + ' the best object size were: \n obj: ' + str(
        best_object_size) + '\n hole: ' + str(
        best_hole_size))

    return best_object_size, best_hole_size


def val_score(model_name, use_metric, iter=10, th=c.val['th_nn'], cl_th=0.8, pp_th=0., return_full=False, obj_size=18, holes_size=3):
    """
    Method to quantify the overall performance of a model
    :param iter: the more iterations the more accurate
    :param th: threshold of the clustering method
    :param model_name: model that should be evaluated
    :background_pred: wether model uses background pred
    :return: Overall Score of the current model
    """
    dtype = torch.float
    device = torch.device('cuda:0')

    dic = data.read_from_json('config/' + str(model_name) + '.json')

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
    model.load_state_dict(torch.load('model/model_weights_' + str(model_name) + '.pt'))
    val_dataset = data.CombinedDataset(corr_path='data/corr/starmy/maxpool/transformed_4/',
                                       corr_sum_folder='data/corr_sum_img/',
                                       sum_folder='data/sum_img/',
                                       mask_folder='data/sum_masks/',
                                       transform=None, device=device, dtype=dtype, test=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0)
    print('Validation loader prepared.')

    set = h.Setup(th_nn=th, model_name=model_name, save_config=False, input_channels=int(dic['input_channels']),
                  embedding_dim=int(dic['embedding_dim']),
                  background_pred=dic['background_pred'] == 'True',
                  nb_iterations=int(dic['nb_iterations']),
                  kernel_bandwidth=dic['kernel_bandwidth'],
                  step_size=float(dic['step_size']),
                  embedding_loss=dic['Embedding Loss'] == 'True',
                  margin=float(dic['margin']),
                  include_background=dic['Include Background'] == 'True',
                  scaling=float(dic['scaling']),
                  subsample_size=int(dic['subsample_size']))
    f1_ = []
    rec_ = []
    prec_ = []
    emb_loss = []
    cel_loss = []

    if return_full:
        for i in range(iter):
            f1_metric, cur_rec, cur_prec, emb_loss_, cel_loss_ = set.validate(val_loader, model, use_metric=use_metric,
                                                                              criterionCEL=nn.CrossEntropyLoss().cuda(),
                                                                              scoring=True,
                                                                              cl_th=cl_th, pp_th=pp_th,
                                                                              return_full=True, holes_size=holes_size,
                                                                              obj_size=obj_size)
            f1_.append(f1_metric)
            rec_.append(cur_rec)
            prec_.append(cur_prec)
            emb_loss.append(emb_loss_)
            cel_loss.append(cel_loss_)
    else:
        for i in range(iter):
            f1_metric, emb_loss_, cel_loss_ = set.validate(val_loader, model, use_metric=use_metric,
                                                           criterionCEL=nn.CrossEntropyLoss().cuda(), scoring=True,
                                                           cl_th=cl_th, pp_th=pp_th, holes_size=holes_size,
                                                           obj_size=obj_size)
            f1_.append(f1_metric)
            emb_loss.append(emb_loss_)
            cel_loss.append(cel_loss_)

    ret_f1 = np.nanmean(f1_)
    ret_emb = np.nanmean(emb_loss)
    ret_cel = np.nanmean(cel_loss)

    if return_full:
        f1_std = np.nanstd(f1_)
        emb_std = np.nanstd(emb_loss)
        cel_std = np.nanstd(cel_loss)
        ret_rec = np.nanmean(rec_)
        rec_std = np.nanstd(rec_)
        ret_prec = np.nanmean(prec_)
        prec_std = np.nanstd(prec_)
        print('Average Val Score of model ' + str(model_name) + ':\t' + str(ret_f1) + '\t' + str(f1_std) +
              '\nAverage Recall\t' + str(ret_rec) + '\t' + str(rec_std) +
              '\nAverage Precision\t' + str(ret_prec) + '\t' + str(prec_std) +
              '\nAverage Emb Loss\t' + str(ret_emb) + '\t' + str(emb_std) +
              '\nAverage Cel Loss\t' + str(ret_cel) + '\t' + str(cel_std))
        return ret_f1, f1_std, ret_rec, rec_std, ret_prec, prec_std, ret_emb, emb_std, ret_cel, cel_std
    else:
        print('Average Val Score of model ' + str(model_name) + ':\t' + str(ret_f1) + '\n '
                'Average Emb Score/Average CEL Score: \t' + str(ret_emb) + '/' + str(ret_cel))
        return ret_f1, ret_emb, ret_cel


def test_th(model_name, background_pred, np_arange=(0.005, 2.05, 0.005), iter=10):
    dtype = torch.float
    device = torch.device('cuda:0')
    model = n.UNetMS(use_background_pred=background_pred)

    model.to(device)
    model.type(dtype)
    model.load_state_dict(torch.load('model/model_weights_' + str(model_name) + '.pt'))
    val_dataset = data.CombinedDataset(corr_path='data/corr/starmy/sliced/slice_size_100/transformed_4/',
                                       transform=None, device=device, dtype=dtype, test=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0)
    print('Validation loader prepared.')

    list = np.arange(np_arange[0], np_arange[1], np_arange[2])
    f1_list = []
    f1_ind_list = []
    for th in list:
        f1_ = []
        set = h.Setup(th_nn=th, save_config=False)
        print('Threshold = ' + str(th))
        for i in range(iter):
            f1_metric = set.validate(val_loader, model)
            f1_.append(f1_metric)
        f1_list.append(sum(f1_)/f1_.__len__())
        f1_ind_list.append(th)
    print('Best Possible Th:\t', np.array(f1_ind_list)[np.argmax(np.array(f1_list))], max(f1_list))
    return np.array(f1_list), np.array(f1_ind_list)


def det_bandwidth(model_name):
    dtype = torch.float
    device = torch.device('cuda:0')

    dic = data.read_from_json('config/' + str(model_name) + '.json')

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
    model.load_state_dict(torch.load('model/model_weights_' + str(model_name) + '.pt'))
    val_dataset = data.CombinedDataset(corr_path='data/corr/starmy/maxpool/transformed_4/',
                                       corr_sum_folder='data/corr_sum_img/',
                                       sum_folder='data/sum_img/',
                                       mask_folder='data/sum_masks/',
                                       transform=None, device=device, dtype=dtype, test=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0)
    print('Validation loader prepared.')

    set = h.Setup(th_nn=0.8, model_name=model_name, save_config=False, input_channels=int(dic['input_channels']),
                  embedding_dim=int(dic['embedding_dim']),
                  background_pred=dic['background_pred'] == 'True',
                  nb_iterations=int(dic['nb_iterations']),
                  kernel_bandwidth=dic['kernel_bandwidth'],
                  step_size=float(dic['step_size']),
                  embedding_loss=dic['Embedding Loss'] == 'True',
                  margin=float(dic['margin']),
                  include_background=dic['Include Background'] == 'True',
                  scaling=float(dic['scaling']),
                  subsample_size=int(dic['subsample_size']))

    model.eval()
    model.MS.val = True

    for i, batch in enumerate(val_loader):
        input = batch['image']
        label = batch['label']

        output, val_loss, y = model(input, label)

        np_label = h.get_diff_labels(label.detach().cpu().numpy())

        plt.imshow(np_label[0])
        plt.show()

        for b in np.unique(np_label[0]):
            print(b)
            if b != 0:
                ind = np.argwhere(np_label[0] == b)
                print(ind.shape, output.size())
                print(output[0, :, ind[:, 0], ind[:, 1]].size())
                emb_vectors = output[0, :, ind[:, 0], ind[:, 1]].t()
                dist = torch.pdist(emb_vectors)
                dist_ = torch.mean(dist)
                print(dist_)

def zero_to_one(input, std=None):
    if std is None:
        return (input - min(input)) / (max(input) - min(input))
    else:
        return (input - min(input)) / (max(input) - min(input)), std / (max(input) - min(input))

